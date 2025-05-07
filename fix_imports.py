import sys

from bowler import LN, Capture, Filename, Query
from fissix.fixer_util import FromImport, Name
from fissix.pygram import python_symbols as syms
from fissix.pytree import Node


def is_import_name(node: LN, capture: Capture, filename: Filename) -> bool:
    return node.type == syms.import_name


def is_import_from(node: LN, capture: Capture, filename: Filename) -> bool:
    return "module_import" in capture or node.type == syms.import_from


def convert_to_import_from(node: LN, capture: Capture, filename: Filename) -> Node:
    module_name = capture["module_name"]
    new_node = FromImport("GPT_SoVITS", [Name(module_name.value, prefix=" ")])
    return new_node


def fix_imports(path_to_top_level_module: str, module_name: str):
    """
    Fix imports in the specified module by renaming them to include .text suffix.

    Args:
        path_to_top_level_module (str): Path to the top-level module directory
        module_name (str): Name of the module to fix imports for
    """
    try:
        (
            Query(path_to_top_level_module)
            .select_module(module_name)
            .filter(is_import_name)
            .modify(convert_to_import_from)
            .execute()
        )
        (
            Query(path_to_top_level_module)
            .select_module(module_name)
            .filter(is_import_from)
            .rename(f"{path_to_top_level_module}.{module_name}")
            .execute()
        )
        print(f"Successfully fixed imports for module: {module_name}")
    except Exception as e:
        print(f"Error fixing imports: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    path = "GPT_SoVITS"
    modules = [
        "AR",
        "BigVGAN",
        "configs",
        "f5_tts",
        "feature_extractor",
        "module",
        "text",
        "TTS_infer_pack",
        "inference_webui",
        "inference_webui_fast",
        "inference_cli",
        "inference_gui",
        "onnx_export",
        "process_ckpt",
        "s1_train",
        "s2_train",
        "s2_train_v3",
        "s2_train_v3_lora",
        "utils",
    ]
    for module in modules:
        fix_imports(path, module)


if __name__ == "__main__":
    main()
