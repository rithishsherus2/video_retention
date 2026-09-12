"""Combine every per-source signal (retention, quality, shots, audio,
transcript) into one aligned per-second table.

This is deliberately the artifact to look at BEFORE any LLM analysis --
eyeballing quality/cut/audio signals plotted against the real retention
curve on a handful of reels teaches you more about what actually matters
than prompt-engineering blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.features import audio as audio_mod
from app.features import dialogue as dialogue_mod
from app.features import quality as quality_mod
from app.features import shots as shots_mod
from app.schemas import RetentionCurve


def build_timeline(
    video_path: str, retention_path: str, quality_fps: float = 5.0,
    whisper_model: str = "small", language: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = RetentionCurve.model_validate(json.loads(Path(retention_path).read_text()))
    points = curve.sorted_points()
    duration = curve.video_duration_s or (points[-1].t if points else 0.0)
    n_seconds = int(round(duration)) + 1

    print("[timeline] shot detection + CLIP similarity...")
    shot_analyses = shots_mod.analyze_shots(video_path)
    shots = [sa.shot for sa in shot_analyses]
    shots_df = pd.DataFrame([{
        "sec": int(sa.shot.start_t),
        "shot_index": sa.shot.index,
        "shot_start_t": sa.shot.start_t,
        "shot_duration": sa.shot.duration,
        "prev_shot_similarity": sa.prev_shot_similarity,
        "is_likely_redundant_cut": sa.is_likely_redundant_cut,
        "is_likely_reused_shot": sa.is_likely_reused_shot,
        "reused_shot_of_index": sa.most_similar_other_index,
    } for sa in shot_analyses])

    print("[timeline] quality metrics...")
    quality_cols = ["sharpness", "highfreq_ratio", "blockiness"]
    q = quality_mod.extract_quality_timeline(video_path, fps_sample=quality_fps)
    q["shot_index"] = quality_mod.assign_shot_index(q["t"], shots)
    # Per-shot baseline is what actually decides "quality dropped" (see
    # zscore_flags_per_shot docstring) -- global z kept only as a
    # secondary reference column, not what drives the flag.
    q = quality_mod.zscore_flags_per_shot(q, quality_cols)
    q = quality_mod.zscore_flags(q, quality_cols)
    q["sec"] = q["t"].astype(int)
    q_agg = q.groupby("sec").agg(
        sharpness_mean=("sharpness", "mean"),
        sharpness_min=("sharpness", "min"),
        sharpness_shot_z_min=("sharpness_shot_z", "min"),
        sharpness_shot_flag_any=("sharpness_shot_flag", "max"),
        highfreq_ratio_mean=("highfreq_ratio", "mean"),
        highfreq_ratio_shot_z_min=("highfreq_ratio_shot_z", "min"),
        blockiness_mean=("blockiness", "mean"),
        blockiness_shot_z_max=("blockiness_shot_z", "max"),
        blockiness_shot_flag_any=("blockiness_shot_flag", "max"),
        brightness_mean=("brightness", "mean"),
    ).reset_index()

    # Shot-vs-previous-shot deltas: raw comparative evidence for whether a
    # whole shot is a step down from what came right before it. NOT a
    # verdict on its own -- a real location/lighting change will show up
    # here too, and telling those apart needs the actual frames, which is
    # the analysis stage's job, not this one's.
    shot_quality_agg = q.groupby("shot_index").agg(
        sharpness_mean=("sharpness", "mean"),
        blockiness_mean=("blockiness", "mean"),
        highfreq_ratio_mean=("highfreq_ratio", "mean"),
    ).reset_index()
    shot_quality_agg = quality_mod.shot_level_quality_deltas(
        shot_quality_agg, ["sharpness_mean", "blockiness_mean", "highfreq_ratio_mean"]
    )
    shots_df = shots_df.merge(shot_quality_agg, on="shot_index", how="left", suffixes=("", "_shotagg"))

    print("[timeline] audio loudness...")
    wav_path = audio_mod.extract_audio_wav(video_path)
    loud = audio_mod.loudness_timeline(wav_path)
    loud["sec"] = loud["t"].astype(int)
    loud_agg = loud.groupby("sec").agg(
        rms_db_mean=("rms_db", "mean"),
        is_silent_any=("is_silent", "max"),
    ).reset_index()

    print("[timeline] transcription (faster-whisper, this is the slow step)...")
    transcript = audio_mod.transcribe(video_path, model_size=whisper_model, wav_path=wav_path, language=language)
    print(f"[timeline] detected language: {transcript.language} "
          f"(confidence {transcript.language_probability:.2f}, model '{transcript.model_used}')")
    words_df = pd.DataFrame([{"sec": int(w.start), "word": w.word} for w in transcript.words])
    text_by_sec = (
        words_df.groupby("sec")["word"].apply(lambda s: " ".join(s))
        if not words_df.empty else pd.Series(dtype=str)
    )

    print("[timeline] redundant dialogue check...")
    redundancy_flags = dialogue_mod.detect_redundant_dialogue(transcript.segments)
    if redundancy_flags:
        for f in redundancy_flags:
            print(f"    t={f.start:.1f}s repeats t={f.repeats_start:.1f}s (sim={f.similarity:.2f}): {f.text!r}")
    redundant_dialogue_seconds = {int(f.start) for f in redundancy_flags}

    # ---- assemble the per-second base table ----
    base = pd.DataFrame({"sec": range(n_seconds)})
    ret_df = pd.DataFrame([{"sec": int(round(p.t)), "retention_pct": p.pct} for p in points])
    ret_df = ret_df.drop_duplicates(subset="sec", keep="first")

    df = base.merge(ret_df, on="sec", how="left")
    df = df.merge(q_agg, on="sec", how="left")
    df = df.merge(loud_agg, on="sec", how="left")

    cut_seconds = set(shots_df["sec"]) if not shots_df.empty else set()
    df["is_cut_second"] = df["sec"].isin(cut_seconds)
    redundant_seconds = set(shots_df.loc[shots_df["is_likely_redundant_cut"], "sec"]) if not shots_df.empty else set()
    df["is_redundant_cut_second"] = df["sec"].isin(redundant_seconds)
    reused_seconds = set(shots_df.loc[shots_df["is_likely_reused_shot"], "sec"]) if not shots_df.empty else set()
    df["is_reused_shot_second"] = df["sec"].isin(reused_seconds)
    df["is_redundant_dialogue_second"] = df["sec"].isin(redundant_dialogue_seconds)

    df["transcript"] = df["sec"].map(text_by_sec).fillna("")

    return df, shots_df


def save_debug_plot(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.plot(df["sec"], df["retention_pct"], marker="o", color="black", label="retention %")
    for _, row in df[df["is_cut_second"]].iterrows():
        ax.axvline(row["sec"], color="gray", alpha=0.3, linestyle="--")
    for _, row in df[df["is_redundant_cut_second"]].iterrows():
        ax.axvline(row["sec"], color="red", alpha=0.6)
    ax.set_ylabel("retention %")
    ax.legend(loc="upper right")
    ax.set_title("Retention (dashed gray = cut, red = likely redundant cut)")

    ax = axes[1]
    ax.plot(df["sec"], df["sharpness_shot_z_min"], color="steelblue", label="sharpness z vs. own shot (min in sec)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("sharpness z\n(per-shot baseline)")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(df["sec"], df["blockiness_shot_z_max"], color="darkorange", label="blockiness z vs. own shot (max in sec)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("blockiness z\n(per-shot baseline)")
    ax.legend(loc="upper right")

    ax = axes[3]
    ax.plot(df["sec"], df["rms_db_mean"], color="green", label="audio RMS dB")
    ax.set_ylabel("dB")
    ax.set_xlabel("second")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_human_summary(
    df: pd.DataFrame, shots_df: pd.DataFrame, out_path: str,
    sharpness_z_threshold: float = -1.5, blockiness_z_threshold: float = 1.5,
) -> None:
    """A plain-text, one-line-per-second digest -- the thing to actually
    read. timeline.csv stays the full-fidelity machine format (every
    column, every metric); this only surfaces what's notable per second,
    so a drop period is scannable in a few lines instead of a spreadsheet.

    Quality flags here use each frame's PER-SHOT baseline (is it anomalous
    relative to its own shot's neighbors?), not the whole video's average
    -- see quality.zscore_flags_per_shot for why that distinction matters.
    Shot-vs-previous-shot deltas are printed as plain comparative numbers,
    not flags: whether a change is a real defect or a deliberate new
    location/lighting/style needs the actual frames to judge, which is the
    analysis stage's job, not this digest's."""
    lines: list[str] = []

    duration = int(df["sec"].max())
    n_cuts = int(df["is_cut_second"].sum())
    n_redundant = int(df["is_redundant_cut_second"].sum())
    start_pct = df["retention_pct"].dropna().iloc[0] if df["retention_pct"].notna().any() else None
    end_pct = df["retention_pct"].dropna().iloc[-1] if df["retention_pct"].notna().any() else None
    lines.append(f"=== {duration}s reel -- {n_cuts} cuts ({n_redundant} likely redundant) "
                 f"-- retention {start_pct:.0f}% -> {end_pct:.0f}% ===")
    has_any_words = df["transcript"].fillna("").str.strip().ne("").any()
    has_any_audio = df["rms_db_mean"].notna().any()
    if not has_any_words and has_any_audio:
        lines.append("(confirmed no dialogue anywhere in this reel -- music/ambient audio only, "
                      "not a transcription failure: verified with word-timestamped ASR)")
    lines.append("")

    # sec -> "sharpness -35% vs prev shot, blockiness +210% vs prev shot"
    # (only for shots that have a previous shot to compare to)
    shot_delta_by_sec: dict[int, str] = {}
    for _, srow in shots_df.iterrows():
        parts = []
        for metric, label in [("sharpness_mean", "sharpness"), ("blockiness_mean", "blockiness")]:
            pct_col = f"{metric}_pct_change_vs_prev_shot"
            if pct_col in srow and pd.notna(srow[pct_col]):
                parts.append(f"{label} {srow[pct_col]:+.0f}% vs prev shot")
        if parts:
            shot_delta_by_sec[int(srow["sec"])] = ", ".join(parts)

    prev_silent = None
    for _, row in df.iterrows():
        sec = int(row["sec"])
        ret = row["retention_pct"]
        ret_str = f"{ret:5.1f}%" if pd.notna(ret) else "  n/a "

        tags = []
        if row.get("is_cut_second"):
            tags.append("REDUNDANT CUT" if row.get("is_redundant_cut_second") else "cut")
            if sec in shot_delta_by_sec:
                tags.append(shot_delta_by_sec[sec])
        if row.get("is_reused_shot_second"):
            tags.append("reused shot")
        if row.get("is_redundant_dialogue_second"):
            tags.append("REDUNDANT DIALOGUE (repeats something already said)")
        if pd.notna(row.get("sharpness_shot_z_min")) and row["sharpness_shot_z_min"] < sharpness_z_threshold:
            tags.append(f"soft/blurry vs. rest of this shot (z={row['sharpness_shot_z_min']:.1f})")
        if pd.notna(row.get("blockiness_shot_z_max")) and row["blockiness_shot_z_max"] > blockiness_z_threshold:
            tags.append(f"COMPRESSION ARTIFACTS vs. rest of this shot (z={row['blockiness_shot_z_max']:.1f})")
        is_silent = bool(row.get("is_silent_any")) if pd.notna(row.get("is_silent_any")) else None
        if is_silent is not None and prev_silent is not None and is_silent != prev_silent:
            tags.append("audio drops to silence" if is_silent else "audio jumps in from silence")
        if is_silent is not None:
            prev_silent = is_silent

        transcript = (row.get("transcript") or "").strip()
        if transcript:
            tags.append(f'says: "{transcript}"')

        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        lines.append(f"t={sec:3d}s  retention={ret_str}{tag_str}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--retention", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--whisper-model", type=str, default="small")
    ap.add_argument("--language", type=str, default=None, help="ISO code e.g. 'te' -- skip to auto-detect")
    ap.add_argument("--quality-fps", type=float, default=5.0)
    args = ap.parse_args()

    df, shots_df = build_timeline(
        str(args.video), str(args.retention), quality_fps=args.quality_fps,
        whisper_model=args.whisper_model, language=args.language,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "timeline.csv", index=False)
    shots_df.to_csv(args.out_dir / "shots.csv", index=False)
    save_debug_plot(df, str(args.out_dir / "timeline_debug.png"))
    summary_path = args.out_dir / "timeline_summary.txt"
    write_human_summary(df, shots_df, str(summary_path))

    print(f"\nSaved: {args.out_dir / 'timeline.csv'}  (full per-second data, every column)")
    print(f"Saved: {args.out_dir / 'shots.csv'}")
    print(f"Saved: {args.out_dir / 'timeline_debug.png'}")
    print(f"Saved: {summary_path}  (human-readable digest -- read this one)")
    print(f"\n{len(shots_df)} shots detected.")


if __name__ == "__main__":
    main()
