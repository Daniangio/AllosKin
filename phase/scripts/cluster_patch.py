from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

from phase.services.project_store import ProjectStore
from phase.workflows.clustering import (
    confirm_cluster_residue_patch,
    create_cluster_residue_patch,
    discard_cluster_residue_patch,
    list_cluster_patches,
)


def _make_store(root: str | None) -> ProjectStore:
    if root and str(root).strip():
        return ProjectStore(base_dir=Path(root).expanduser().resolve() / "projects")
    return ProjectStore()


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _split_int_csv(value: str | None) -> List[int]:
    out: List[int] = []
    for token in _split_csv(value):
        try:
            out.append(int(token))
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid integer token in list: {token!r}") from exc
    return out


def _print_patch_rows(rows: Iterable[dict], *, pipe: bool = False) -> None:
    for row in rows:
        patch_id = str(row.get("patch_id") or "")
        name = str(row.get("name") or patch_id)
        status = str(row.get("status") or "preview")
        created_at = str(row.get("created_at") or "")
        residues = ",".join([str(v) for v in (row.get("residue_keys") or [])])
        if pipe:
            print(f"{patch_id}|{name}|{status}|{residues}|{created_at}")
        else:
            print(f"- {name} ({patch_id})")
            print(f"    status: {status}")
            if residues:
                print(f"    residues: {residues}")
            if created_at:
                print(f"    created_at: {created_at}")


def cmd_list(args: argparse.Namespace) -> int:
    store = _make_store(args.root)
    out = list_cluster_patches(
        args.project_id,
        args.system_id,
        args.cluster_id,
        store=store,
    )
    patches = out.get("patches") or []
    if args.pipe:
        _print_patch_rows(patches, pipe=True)
    else:
        if not patches:
            print("(no patches)")
        else:
            _print_patch_rows(patches, pipe=False)
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    residue_keys = _split_csv(args.residue_keys)
    residue_indices = _split_int_csv(args.residue_indices)
    if not residue_keys and not residue_indices:
        raise SystemExit("Provide --residue-keys or --residue-indices.")
    if str(args.cluster_selection_mode).lower() == "inconsistent" and args.inconsistent_threshold is None:
        raise SystemExit("--inconsistent-threshold is required when --cluster-selection-mode=inconsistent.")

    store = _make_store(args.root)
    out = create_cluster_residue_patch(
        args.project_id,
        args.system_id,
        args.cluster_id,
        residue_indices=residue_indices or None,
        residue_keys=residue_keys or None,
        n_clusters=args.n_clusters,
        cluster_selection_mode=args.cluster_selection_mode,
        inconsistent_threshold=args.inconsistent_threshold,
        inconsistent_depth=args.inconsistent_depth,
        linkage_method=args.linkage_method,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        halo_percentile=args.halo_percentile,
        max_cluster_frames=args.max_cluster_frames,
        patch_name=args.name,
        store=store,
    )
    patch = out.get("patch") or {}
    print(f"created patch_id={out.get('patch_id')}")
    print(f"name={patch.get('name')}")
    print(f"residues={','.join((patch.get('residues') or {}).get('keys') or [])}")
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    store = _make_store(args.root)
    out = confirm_cluster_residue_patch(
        args.project_id,
        args.system_id,
        args.cluster_id,
        patch_id=args.patch_id,
        recompute_assignments=not args.no_recompute,
        store=store,
    )
    print(f"confirmed patch_id={out.get('patch_id')}")
    if args.no_recompute:
        print("recompute_assignments=false")
    else:
        assignments = out.get("assignments") or {}
        samples = assignments.get("samples") or []
        print(f"md_samples_recomputed={len(samples)}")
    return 0


def cmd_discard(args: argparse.Namespace) -> int:
    store = _make_store(args.root)
    out = discard_cluster_residue_patch(
        args.project_id,
        args.system_id,
        args.cluster_id,
        patch_id=args.patch_id,
        store=store,
    )
    print(f"discarded patch_id={out.get('patch_id')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Preview/confirm/discard residue-level cluster patches."
    )
    ap.add_argument(
        "--root",
        default=os.getenv("PHASE_DATA_ROOT", ""),
        help="Data root containing projects/ (default: $PHASE_DATA_ROOT).",
    )
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)

    sp = ap.add_subparsers(dest="cmd", required=True)

    p_list = sp.add_parser("list", help="List stored preview patches.")
    p_list.add_argument("--pipe", action="store_true", help="Emit machine-readable rows.")
    p_list.set_defaults(func=cmd_list)

    p_create = sp.add_parser("create", help="Create a preview patch.")
    p_create.add_argument("--residue-keys", default="", help="Comma-separated residue keys.")
    p_create.add_argument("--residue-indices", default="", help="Comma-separated residue indices.")
    p_create.add_argument("--n-clusters", type=int, default=None)
    p_create.add_argument(
        "--cluster-selection-mode",
        default="maxclust",
        choices=["maxclust", "inconsistent"],
        help="How flat clusters are chosen from hierarchical linkage.",
    )
    p_create.add_argument(
        "--inconsistent-threshold",
        type=float,
        default=None,
        help="Threshold t used with criterion='inconsistent'.",
    )
    p_create.add_argument(
        "--inconsistent-depth",
        type=int,
        default=2,
        help="Depth parameter used with criterion='inconsistent'.",
    )
    p_create.add_argument("--linkage-method", default="ward", choices=["ward", "complete", "average", "single"])
    p_create.add_argument("--covariance-type", default="full", choices=["full", "diag"])
    p_create.add_argument("--reg-covar", type=float, default=1e-5)
    p_create.add_argument("--halo-percentile", type=float, default=5.0)
    p_create.add_argument("--max-cluster-frames", type=int, default=None)
    p_create.add_argument("--name", default="", help="Optional patch display name.")
    p_create.set_defaults(func=cmd_create)

    p_confirm = sp.add_parser("confirm", help="Confirm a patch and swap it into canonical cluster labels.")
    p_confirm.add_argument("--patch-id", required=True)
    p_confirm.add_argument("--no-recompute", action="store_true", help="Skip re-evaluating MD samples.")
    p_confirm.set_defaults(func=cmd_confirm)

    p_discard = sp.add_parser("discard", help="Discard a preview patch.")
    p_discard.add_argument("--patch-id", required=True)
    p_discard.set_defaults(func=cmd_discard)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
