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

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.features import audio as audio_mod
from app.features import dialogue as dialogue_mod
from app.features import quality as quality_mod
from app.features import shots as shots_mod
from app.schemas import RetentionCurve


def _video_duration_s(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def build_timeline(
    video_path: str, retention_path: str | None = None, quality_fps: float = 5.0,
    whisper_model: str = "small", language: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """retention_path is optional: pass None to build the same feature
    timeline for a reel that has no retention recording at all (e.g. an
    "ideal" reel being decoded for its rules, or a test reel evaluated
    without one) -- retention_pct just stays NaN throughout and duration
    is read from the video file itself instead of the curve."""
    if retention_path:
        curve = RetentionCurve.model_validate(json.loads(Path(retention_path).read_text()))
        points = curve.sorted_points()
        duration = curve.video_duration_s or (points[-1].t if points else 0.0)
    else:
        points = []
        duration = _video_duration_s(video_path)
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
    if points:
        ret_df = pd.DataFrame([{"sec": int(round(p.t)), "retention_pct": p.pct} for p in points])
        ret_df = ret_df.drop_duplicates(subset="sec", keep="first")
    else:
        ret_df = pd.DataFrame(columns=["sec", "retention_pct"])

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


def compute_reel_features(df: pd.DataFrame, shots_df: pd.DataFrame, duration_s: float) -> dict:
    """Deterministic, no-LLM aggregate stats describing a reel's pacing and
    technical execution as a whole (as opposed to build_timeline's
    per-second table). This is the raw material fed into Gemini's
    decode/critique prompts in app.analysis.rules/evaluate -- it doesn't
    itself judge whether any of these numbers are good or bad."""
    shot_durations = shots_df["shot_duration"].tolist() if not shots_df.empty else []
    n_shots = len(shots_df)
    cuts_in_first_3s = int((shots_df["shot_start_t"] < 3.0).sum()) if not shots_df.empty else 0

    transcript_col = df["transcript"].fillna("") if "transcript" in df else pd.Series(dtype=str)
    dialogue_seconds = int(transcript_col.str.strip().ne("").sum())
    full_transcript = " ".join(t for t in transcript_col if t).strip()

    silent_seconds = int(df["is_silent_any"].fillna(False).sum()) if "is_silent_any" in df else 0
    redundant_cuts = int(df["is_redundant_cut_second"].fillna(False).sum()) if "is_redundant_cut_second" in df else 0
    reused_shots = int(df["is_reused_shot_second"].fillna(False).sum()) if "is_reused_shot_second" in df else 0
    redundant_dialogue = int(df["is_redundant_dialogue_second"].fillna(False).sum()) if "is_redundant_dialogue_second" in df else 0

    retention_stats = _retention_summary_stats(df, duration_s)
    reused_shot_instances = _reused_shot_instances(shots_df)

    return {
        "duration_s": round(duration_s, 1),
        "shot_count": n_shots,
        "avg_shot_duration_s": round(sum(shot_durations) / len(shot_durations), 2) if shot_durations else None,
        "min_shot_duration_s": round(min(shot_durations), 2) if shot_durations else None,
        "max_shot_duration_s": round(max(shot_durations), 2) if shot_durations else None,
        "cuts_per_10s": round(n_shots / duration_s * 10, 2) if duration_s else None,
        "cuts_in_first_3s": cuts_in_first_3s,
        "has_dialogue": dialogue_seconds > 0,
        "dialogue_seconds": dialogue_seconds,
        "dialogue_density": round(dialogue_seconds / duration_s, 2) if duration_s else None,
        "silent_seconds": silent_seconds,
        "redundant_cuts": redundant_cuts,
        "reused_shots": reused_shots,
        "reused_shot_instances": reused_shot_instances,
        "redundant_dialogue_moments": redundant_dialogue,
        "avg_sharpness": round(float(df["sharpness_mean"].mean()), 1) if "sharpness_mean" in df and df["sharpness_mean"].notna().any() else None,
        "avg_blockiness": round(float(df["blockiness_mean"].mean()), 3) if "blockiness_mean" in df and df["blockiness_mean"].notna().any() else None,
        "avg_brightness": round(float(df["brightness_mean"].mean()), 1) if "brightness_mean" in df and df["brightness_mean"].notna().any() else None,
        "avg_audio_rms_db": round(float(df["rms_db_mean"].mean()), 1) if "rms_db_mean" in df and df["rms_db_mean"].notna().any() else None,
        "full_transcript": full_transcript,
        **retention_stats,
    }


def _reused_shot_instances(shots_df: pd.DataFrame) -> list[dict]:
    """Each shot flagged as a near-duplicate of an earlier/later non-adjacent
    shot (see app.features.shots.analyze_shots), with real timestamps for
    both this occurrence and the shot it matches -- not just the aggregate
    count reused_shots already gives. This is what lets the UI point at the
    actual reused footage instead of just saying "some shots were reused"."""
    if shots_df.empty or "is_likely_reused_shot" not in shots_df:
        return []
    start_by_index = dict(zip(shots_df["shot_index"], shots_df["shot_start_t"]))
    instances = []
    for _, row in shots_df[shots_df["is_likely_reused_shot"].fillna(False)].iterrows():
        start_t = float(row["shot_start_t"])
        other_idx = row.get("reused_shot_of_index")
        other_start = start_by_index.get(other_idx) if pd.notna(other_idx) else None
        instances.append({
            "start_t": round(start_t, 1),
            "end_t": round(start_t + float(row["shot_duration"]), 1),
            "reused_of_start_t": round(float(other_start), 1) if other_start is not None else None,
        })
    return instances


def _retention_summary_stats(df: pd.DataFrame, duration_s: float) -> dict:
    """Only populated when this reel's OWN retention curve was supplied
    (an ideal reel's retention recording is optional -- see
    app.analysis.rules.extract_rules_from_ideal_reels). Gives the rule-
    synthesis step actual evidence for which techniques correlate with
    viewers still watching at a given second, not just an aggregate
    engagement count for the whole reel."""
    if "retention_pct" not in df or not df["retention_pct"].notna().any():
        return {"has_retention_data": False}

    ret = df[["sec", "retention_pct"]].dropna().sort_values("sec")

    def _at(target_sec: float):
        at_or_before = ret[ret["sec"] <= target_sec]
        row = at_or_before.iloc[-1] if not at_or_before.empty else ret.iloc[0]
        return round(float(row["retention_pct"]), 1)

    return {
        "has_retention_data": True,
        "retention_at_3s_pct": _at(3) if duration_s >= 1 else None,
        "retention_at_5s_pct": _at(5) if duration_s >= 3 else None,
        "retention_at_10s_pct": _at(10) if duration_s >= 6 else None,
        "retention_end_pct": round(float(ret.iloc[-1]["retention_pct"]), 1),
        "retention_min_pct": round(float(ret["retention_pct"].min()), 1),
    }


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
    has_retention = df["retention_pct"].notna().any()
    start_pct = df["retention_pct"].dropna().iloc[0] if has_retention else None
    end_pct = df["retention_pct"].dropna().iloc[-1] if has_retention else None
    retention_str = f"retention {start_pct:.0f}% -> {end_pct:.0f}%" if has_retention else "no retention data"
    lines.append(f"=== {duration}s reel -- {n_cuts} cuts ({n_redundant} likely redundant) -- {retention_str} ===")
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
