"""Freeze outcome-independent dense WSI coordinates from verified original TIFFs.

The thumbnail is streamed from standalone JPEG TIFF tiles with Pillow, so no
imagecodecs, OpenSlide, or full-resolution image allocation is required. Requires
numpy, Pillow, tifffile, and opencv-python-headless. See dense_plan.md for the
intentional, documented differences from the unrecovered historical tile set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import PIL
import tifffile

try:
    import cv2
except ImportError as exc:
    raise ImportError("Install opencv-python-headless to build the dense manifest") from exc


CONFIG = {
    "protocol": "A5_DENSE_NEW_RUN_V1",
    "source_level": 0,
    "crop_size": 512,
    "contour_footprint": 512,
    "stride": 256,
    "thumbnail_downsample": 64,
    "thumbnail_resample": "Pillow_BOX_each_aligned_TIFF_tile",
    "partial_thumbnail_block": "white_pad_to_64_pixel_block",
    "hsv_saturation_threshold": 8,
    "median_kernel": 7,
    "closing_kernel": 4,
    "otsu": False,
    "contour_retrieval": "RETR_CCOMP",
    "contour_chain": "CHAIN_APPROX_NONE",
    "reference_patch_size": 512,
    "foreground_area_multiplier": 100,
    "hole_area_multiplier": 16,
    "maximum_holes_per_contour": 8,
    "contour_rule": "four_pt_easy",
    "four_point_center_shift": 0.5,
    "hole_rule": "exclude_if_patch_center_strictly_inside_retained_hole",
    "grid_origin": "each_tissue_contour_level0_bounding_box",
    "use_padding": True,
    "padding_rgb": [255, 255, 255],
    "duplicate_coordinates": "deduplicate_by_source_x_y",
    "post_manifest_hsv_filter": False,
    "outcomes_used": False,
    "model_outputs_used": False,
}
FIELDS = [
    "case_id", "outer_fold", "source_sha256", "source_path", "x", "y",
    "crop_size", "level", "source_width", "source_height", "pad_right",
    "pad_bottom", "config_sha256",
]
CONDITIONS = (
    "30-30-120-N", "30-40-80-N", "36-20-160-A", "36-30-120-A",
    "36-30-120-N", "36-40-80-A", "36-40-80-N",
)
EXPECTED_CASES = {f"{condition}-{rep}" for condition in CONDITIONS for rep in (1, 2, 3)}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_path(value):
    return str(value).replace("\\", "/").rstrip("/").casefold()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_sources(audit_path, cases_path):
    """Only join identity fields; never read outcome columns into processing."""
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8-sig"))
    if audit.get("summary", {}).get("accepted") is not True:
        raise ValueError("The source copy acceptance audit is not accepted")
    files = audit["files"]
    if len(files) != 32:
        raise ValueError(f"Expected the verified 32-source audit, found {len(files)}")
    by_mac, by_destination = {}, {}
    for record in files:
        source_hash = str(record.get("source_sha256", "")).lower()
        if (record.get("passed") is not True or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                or source_hash != str(record.get("destination_sha256", "")).lower()):
            raise ValueError(f"Invalid source acceptance record: {record.get('destination')}")
        for lookup, key in ((by_mac, normalized_path(record["source"])),
                            (by_destination, normalized_path(record["destination"]))):
            if key in lookup:
                raise ValueError(f"Ambiguous source audit path: {key}")
            lookup[key] = record
    cases_path = Path(cases_path)
    if cases_path.suffix.lower() == ".json":
        case_rows = json.loads(cases_path.read_text(encoding="utf-8-sig"))["cases"]
    else:
        with cases_path.open(encoding="utf-8-sig", newline="") as stream:
            case_rows = list(csv.DictReader(stream))
    selected = []
    for row in case_rows:
        case_id = row.get("case_id") or row.get("casekey")
        mac_path = row.get("mac_source_tif")
        local_path = row.get("source_path")
        record = by_mac.get(normalized_path(mac_path)) if mac_path else None
        if record is None and local_path:
            record = by_destination.get(normalized_path(local_path))
        if record is None:
            raise ValueError(f"No exact audited source mapping for {case_id}")
        source_hash = record["source_sha256"].lower()
        if row.get("source_sha256") and str(row["source_sha256"]).lower() != source_hash:
            raise ValueError(f"Case/source hash disagrees with acceptance audit: {case_id}")
        if case_id not in EXPECTED_CASES:
            raise ValueError(f"Unexpected case ID: {case_id}")
        source_path = Path(record["destination"]).resolve(strict=True)
        expected_size = int(record["expected_bytes"])
        if source_path.stat().st_size != expected_size:
            raise ValueError(f"Source file size changed: {source_path}")
        selected.append({"case_id": case_id, "outer_fold": int(case_id[-1]),
                         "source_path": str(source_path), "source_sha256": source_hash,
                         "expected_bytes": expected_size})
    if len(selected) != 21 or {r["case_id"] for r in selected} != EXPECTED_CASES:
        raise ValueError("Final source map must contain exactly the prespecified 21 cases")
    if len({r["source_sha256"] for r in selected}) != 21:
        raise ValueError("The final 21 animals do not have 21 unique source WSI hashes")
    return sorted(selected, key=lambda r: r["case_id"])


def decode_jpeg_tile(compressed, expected_size, tables=None):
    """Handle standalone JPEG tiles and TIFF JPEG tables, without imagecodecs."""
    if tables:
        if not (tables.startswith(b"\xff\xd8") and tables.endswith(b"\xff\xd9")
                and compressed.startswith(b"\xff\xd8")):
            raise ValueError("Unsupported TIFF JPEGTables framing")
        compressed = tables[:-2] + compressed[2:]
    with Image.open(io.BytesIO(compressed)) as tile:
        if tile.size != expected_size:
            raise ValueError(f"JPEG tile dimensions {tile.size} != {expected_size}")
        tile.load()
        return tile.convert("RGB")


def ome_pixel_size(description):
    if not description or "<OME" not in description:
        return {}
    try:
        root = ET.fromstring(description)
    except ET.ParseError:
        return {"ome_parse_error": True}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Pixels":
            return {key: element.attrib[key] for key in
                    ("PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY", "PhysicalSizeYUnit")
                    if key in element.attrib}
    return {}


def build_thumbnail(path, progress=None):
    """Fixed64 downsample; at most one decoded JPEG tile plus thumbnail in RAM."""
    scale = CONFIG["thumbnail_downsample"]
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        if (not page.is_tiled or int(page.compression) != 7 or page.samplesperpixel != 3
                or page.bitspersample != 8 or int(page.planarconfig) != 1):
            raise ValueError("Requires tiled JPEG, 8-bit, 3-channel contiguous level0 TIFF")
        orientation = page.tags.get("Orientation")
        if orientation is not None and int(orientation.value) != 1:
            raise ValueError("Unsupported non-top-left TIFF orientation; do not silently rotate")
        width, height = int(page.imagewidth), int(page.imagelength)
        tw, th = int(page.tilewidth), int(page.tilelength)
        if tw % scale or th % scale:
            raise ValueError("TIFF tile dimensions must be multiples of fixed thumbnail downsample64")
        cols, rows = math.ceil(width / tw), math.ceil(height / th)
        if len(page.dataoffsets) != cols * rows or len(page.databytecounts) != cols * rows:
            raise ValueError("Unexpected TIFF tile count/layout")
        thumbnail = Image.new("RGB", (math.ceil(width / scale), math.ceil(height / scale)), "white")
        tables = page.jpegtables
        metadata = {"source_width": width, "source_height": height, "tiff_tile_width": tw,
                    "tiff_tile_height": th, "jpeg_tiles_decoded": cols * rows,
                    "thumbnail_width": thumbnail.width, "thumbnail_height": thumbnail.height,
                    "thumbnail_pixel_scale_x": scale, "thumbnail_pixel_scale_y": scale,
                    "physical_pixel_size": ome_pixel_size(page.description)}
        with Path(path).open("rb") as stream:
            for index, (offset, count) in enumerate(zip(page.dataoffsets, page.databytecounts)):
                if count <= 0:
                    raise ValueError(f"Empty TIFF tile {index}")
                stream.seek(offset)
                compressed = stream.read(count)
                if len(compressed) != count:
                    raise ValueError(f"Truncated TIFF tile {index}")
                tile = decode_jpeg_tile(compressed, (tw, th), tables)
                tx, ty = index % cols, index // cols
                valid_width, valid_height = min(tw, width - tx * tw), min(th, height - ty * th)
                if valid_width != tw or valid_height != th:
                    padded = Image.new("RGB", (tw, th), "white")
                    padded.paste(tile.crop((0, 0, valid_width, valid_height)), (0, 0))
                    tile.close()
                    tile = padded
                small = tile.resize((tw // scale, th // scale), Image.Resampling.BOX)
                thumbnail.paste(small, (tx * tw // scale, ty * th // scale))
                small.close()
                tile.close()
                if progress and ((index + 1) % 4096 == 0 or index + 1 == cols * rows):
                    progress(index + 1, cols * rows)
    return thumbnail, metadata


def segment_thumbnail(thumbnail):
    """CLAM-style tissue segmentation with exact legacy OpenCV operations."""
    rgb = np.asarray(thumbnail.convert("RGB"))
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    median = cv2.medianBlur(saturation, CONFIG["median_kernel"])
    _, mask = cv2.threshold(median, CONFIG["hsv_saturation_threshold"], 255, cv2.THRESH_BINARY)
    kernel = np.ones((CONFIG["closing_kernel"], CONFIG["closing_kernel"]), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    retained = []
    reference_area = int(CONFIG["reference_patch_size"] ** 2 / CONFIG["thumbnail_downsample"] ** 2)
    min_foreground = CONFIG["foreground_area_multiplier"] * reference_area
    min_hole = CONFIG["hole_area_multiplier"] * reference_area
    if hierarchy is not None:
        relations = hierarchy[0]
        for index in np.flatnonzero(relations[:, 3] == -1):
            hole_indices = np.flatnonzero(relations[:, 3] == index)
            area = cv2.contourArea(contours[index]) - sum(cv2.contourArea(contours[i]) for i in hole_indices)
            if area <= min_foreground:
                continue
            # All holes enter area filtering; only the largest qualifying eight
            # enter patch-center exclusion, as in the recovered code.
            holes = sorted((contours[i] for i in hole_indices), key=cv2.contourArea, reverse=True)
            holes = [h for h in holes[:CONFIG["maximum_holes_per_contour"]]
                     if cv2.contourArea(h) > min_hole]
            retained.append((contours[index], holes))
    retained.sort(key=lambda entry: tuple(cv2.boundingRect(entry[0])))
    effective_mask = np.zeros(mask.shape, dtype=np.uint8)
    for contour, holes in retained:
        component = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
        if holes:
            cv2.drawContours(component, holes, -1, 0, cv2.FILLED)
        effective_mask = cv2.bitwise_or(effective_mask, component)
    scale = CONFIG["thumbnail_downsample"]
    level0_contours = [(np.asarray(contour * scale, dtype=np.int32),
                        [np.asarray(hole * scale, dtype=np.int32) for hole in holes])
                       for contour, holes in retained]
    metadata = {"retained_tissue_contours": len(retained),
                "retained_holes": sum(len(holes) for _, holes in retained),
                "minimum_foreground_area_thumbnail_pixels": min_foreground,
                "minimum_hole_area_thumbnail_pixels": min_hole,
                "thumbnail_foreground_pixels": int(np.count_nonzero(effective_mask))}
    return mask, effective_mask, level0_contours, metadata


def candidate_is_tissue(x, y, contour, holes):
    size = CONFIG["contour_footprint"]
    center = (x + size // 2, y + size // 2)
    shift = int(size // 2 * CONFIG["four_point_center_shift"])
    points = [(center[0] - shift, center[1] - shift), (center[0] + shift, center[1] + shift),
              (center[0] + shift, center[1] - shift), (center[0] - shift, center[1] + shift)]
    if not any(cv2.pointPolygonTest(contour, tuple(map(float, point)), False) >= 0 for point in points):
        return False
    return not any(cv2.pointPolygonTest(hole, tuple(map(float, center)), False) > 0 for hole in holes)


def generate_coordinates(contours, width, height):
    coordinates = set()
    for contour, holes in contours:
        left, top, box_width, box_height = cv2.boundingRect(contour)
        for x in range(left, left + box_width, CONFIG["stride"]):
            for y in range(top, top + box_height, CONFIG["stride"]):
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if candidate_is_tissue(x, y, contour, holes):
                    coordinates.add((x, y))
    return sorted(coordinates)


def write_coordinate_csv(path, source, coordinates, dimensions, config_hash):
    width, height = dimensions
    size = CONFIG["crop_size"]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for x, y in coordinates:
            row = {key: source[key] for key in ("case_id", "outer_fold", "source_sha256", "source_path")}
            row.update(x=x, y=y, crop_size=size, level=0, source_width=width, source_height=height,
                       pad_right=max(0, x + size - width), pad_bottom=max(0, y + size - height),
                       config_sha256=config_hash)
            writer.writerow(row)


def build_manifest(audit_path, cases_path, output_dir):
    cv2.setNumThreads(1)
    sources = load_sources(audit_path, cases_path)
    cwd = Path.cwd().resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(cwd) or output_dir == cwd:
        raise ValueError("Output directory must be a new child directory inside the current workspace")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    config_hash = hashlib.sha256(canonical_json(CONFIG)).hexdigest()
    write_json(output_dir / "config.json", {"configuration": CONFIG, "config_sha256": config_hash})
    write_json(output_dir / "input_sources.json", sources)
    software = {"python": sys.version, "numpy": np.__version__, "Pillow": PIL.__version__,
                "opencv": cv2.__version__, "tifffile": tifffile.__version__}
    write_json(output_dir / "software.json", software)
    case_summaries = []
    try:
        for number, source in enumerate(sources, 1):
            case = source["case_id"]
            path = Path(source["source_path"])
            print(f"[{number}/21] {case}: verifying source SHA256", flush=True)
            stat_before = path.stat()
            if sha256_file(path) != source["source_sha256"]:
                raise ValueError(f"Source SHA256 changed: {path}")
            thumbnail, image_metadata = build_thumbnail(
                path, progress=lambda done, total: print(f"  JPEG tiles {done}/{total}", flush=True))
            raw_mask, effective_mask, contours, segmentation_metadata = segment_thumbnail(thumbnail)
            width, height = image_metadata["source_width"], image_metadata["source_height"]
            coordinates = generate_coordinates(contours, width, height)
            stat_after = path.stat()
            if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
                raise ValueError(f"Source file changed during reading: {path}")
            case_dir = output_dir / "cases" / case
            case_dir.mkdir(parents=True)
            thumbnail.save(case_dir / "thumbnail.png")
            thumbnail.close()
            Image.fromarray(raw_mask).save(case_dir / "segmentation_raw.png")
            Image.fromarray(effective_mask).save(case_dir / "tissue_mask.png")
            if not coordinates:
                raise ValueError(f"No tissue coordinates for {case}; inspect saved QC images as a technical failure")
            write_coordinate_csv(case_dir / "coordinates.csv", source, coordinates, (width, height), config_hash)
            summary = {**source, **image_metadata, **segmentation_metadata,
                       "coordinate_count": len(coordinates), "config_sha256": config_hash,
                       "source_hash_verified_this_run": True,
                       "coordinates_sha256": sha256_file(case_dir / "coordinates.csv"),
                       "thumbnail_sha256": sha256_file(case_dir / "thumbnail.png"),
                       "tissue_mask_sha256": sha256_file(case_dir / "tissue_mask.png"),
                       "padded_coordinates": sum(x + CONFIG["crop_size"] > width or
                                                 y + CONFIG["crop_size"] > height for x, y in coordinates)}
            write_json(case_dir / "metadata.json", summary)
            case_summaries.append(summary)
            print(f"  accepted coordinates: {len(coordinates)}", flush=True)
        partial_path = output_dir / "coordinates_all21.csv.partial"
        with partial_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for source in sources:
                with (output_dir / "cases" / source["case_id"] / "coordinates.csv").open(
                        encoding="utf-8", newline="") as case_stream:
                    writer.writerows(csv.DictReader(case_stream))
            stream.flush()
            os.fsync(stream.fileno())
        final_path = output_dir / "coordinates_all21.csv"
        partial_path.replace(final_path)
        write_json(output_dir / "COMPLETE.json", {
            "status": "complete_coordinate_generation_awaiting_visual_qc",
            "case_count": 21, "total_coordinates": sum(s["coordinate_count"] for s in case_summaries),
            "config_sha256": config_hash, "manifest_sha256": sha256_file(final_path),
            "input_audit_sha256": sha256_file(audit_path), "input_cases_sha256": sha256_file(cases_path),
            "generator_sha256": sha256_file(__file__), "software": software, "cases": case_summaries,
            "historical_tile_membership_claimed": False,
            "next_step": "Review thumbnail and mask technical quality; freeze manifest before inference."})
    except Exception as exc:
        write_json(output_dir / "FAILED.json", {"status": "failed", "error": repr(exc),
                                               "completed_cases": [r["case_id"] for r in case_summaries]})
        raise
    print(f"COMPLETE: {final_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-audit", type=Path,
                        default=Path("recovery_audit/verified_mac_wsi_copy_acceptance.json"))
    parser.add_argument("--cases", type=Path, default=Path("recovery_audit/final21_wsi_coverage.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_manifest(args.copy_audit, args.cases, args.output_dir)


if __name__ == "__main__":
    main()
