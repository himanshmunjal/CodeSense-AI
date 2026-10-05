"""
entity_extractor.py
-------------------
PURPOSE:
    This module walks tree-sitter ASTs (produced by tree_sitter_parser.py)
    and extracts structured entities from source code:
        - Functions and methods (with signatures, parameters, return types,
          docstrings, decorators)
        - Classes (with base classes, methods, class-level docstrings)
        - Import statements (normalised to a dependency list)
        - Module-level docstrings and comments

    The output is a rich, typed representation of everything semantically
    meaningful in a source file — the raw material for embedding, indexing,
    and call graph construction.

WHY THIS FILE EXISTS:
    Without entity extraction, CodeSense would have to embed and index entire
    files as single chunks. This has several problems:
        1. A 1,000-line file would produce one massive embedding that averages
           out all the semantics — retrieval would be imprecise.
        2. We couldn't answer "what does function X do?" without scanning the
           whole file.
        3. We couldn't build a call graph (we'd have no function-level identity).
        4. We couldn't give line-precise citations in responses.

    By extracting entities, each function and class becomes its own indexable
    unit with its own embedding, its own metadata (file, lines, language), and
    its own position in the call graph.

WHY AST-BASED EXTRACTION (NOT REGEX):
    Regex can find function definitions in simple cases, but fails on:
        - Nested functions (inner defs inside classes or other functions)
        - Multi-line signatures
        - Decorators before function definitions
        - TypeScript type annotations with complex generics
        - Java annotations

    The AST has exact grammar-level knowledge of where every entity starts
    and ends, what its components are, and how they nest. It is the correct
    tool for this job.

LANGUAGE COVERAGE:
    Each language has its own extraction functions because their AST node
    types differ:
        Python:     function_definition, class_definition, import_statement,
                    import_from_statement, decorated_definition
        JavaScript: function_declaration, arrow_function, class_declaration,
                    import_statement, method_definition
        TypeScript: Same as JS + type_annotation, interface_declaration
        Java:       method_declaration, class_declaration, import_declaration,
                    constructor_declaration, interface_declaration
        Go:         function_declaration, method_declaration (receiver-based,
                    not nested in the type), type_declaration (struct_type /
                    interface_type), import_declaration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from tree_sitter import Node

from parsing.tree_sitter_parser import (
    ParsedFile,
    extract_node_text,
    find_nodes_by_type,
    get_node_line_range,
    iter_children_of_type,
    walk_tree,
)


# ---------------------------------------------------------------------------
# Output data contracts
# ---------------------------------------------------------------------------

@dataclass
class FunctionEntity:
    """
    A single function or method extracted from a source file.

    WHY ALL THESE FIELDS:
        Each field serves a specific downstream purpose:
        - name, signature     → stored in Qdrant metadata for display in results
        - body_text           → the content we embed for semantic search
        - start_line/end_line → used for source citations and line highlighting
        - docstring           → prepended to body_text for richer embeddings
        - parameters          → used for signature-based similarity queries
        - is_method/class_name → needed to build the call graph correctly
                                 (method calls are resolved differently)
        - decorators          → important for Python — a @staticmethod changes
                                how a function is called; @pytest.mark.skip
                                marks it as test code
        - return_type         → aids type-aware search queries
        - calls               → populated by call_graph_builder.py, not here
    """
    name: str
    signature: str
    body_text: str
    relative_file_path: str
    language: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parameters: list[str] = field(default_factory=list)
    return_type: Optional[str] = None
    is_method: bool = False
    class_name: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    calls: list[str] = field(default_factory=list)  # Populated downstream


@dataclass
class ClassEntity:
    """
    A class or interface definition extracted from a source file.

    Attributes:
        name:         Class name.
        base_classes: Inherited classes / implemented interfaces.
        methods:      All method names defined in this class (not full entities —
                      those are in the FunctionEntity list with is_method=True).
        docstring:    Class-level docstring.
        body_text:    Full class source text. Embedded as a secondary unit to
                      support class-level queries ("what does AuthService do?").
        decorators:   Class decorators (e.g. @dataclass in Python,
                      @Injectable in TypeScript/Angular).
        is_interface: True for TypeScript interfaces and Java interfaces.
                      Interfaces have no method bodies; they define contracts.
    """
    name: str
    body_text: str
    relative_file_path: str
    language: str
    start_line: int
    end_line: int
    base_classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_interface: bool = False


@dataclass
class ImportEntity:
    """
    A single import statement extracted from a source file.

    WHY:
        Import statements define the dependency graph at the module level.
        Storing them lets us answer "what external libraries does this file use?"
        and "which files import module X?" — both essential for impact analysis.

    Attributes:
        raw_statement:  The full import statement text.
        module_path:    The module being imported (e.g. "os.path", "react").
        imported_names: Specific names imported ("from X import A, B" → ["A", "B"]).
                        Empty list for whole-module imports.
        is_relative:    True for relative imports (e.g. "from . import utils").
        alias:          Import alias if present ("import numpy as np" → "np").
    """
    raw_statement: str
    module_path: str
    relative_file_path: str
    language: str
    start_line: int
    imported_names: list[str] = field(default_factory=list)
    is_relative: bool = False
    alias: Optional[str] = None


@dataclass
class FileEntities:
    """
    All extracted entities from a single source file.

    This is the top-level output of this module — one FileEntities per file,
    containing all functions, classes, and imports found within it.

    WHY bundle them:
        Downstream consumers (indexing, call graph building) always need all
        three entity types together. A single object avoids returning multiple
        parallel lists that callers must correlate by index.
    """
    source_file_path: str
    language: str
    functions: list[FunctionEntity]
    classes: list[ClassEntity]
    imports: list[ImportEntity]
    module_docstring: Optional[str] = None


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_entities(parsed_file: ParsedFile) -> FileEntities:
    """
    Extract all structured entities from a ParsedFile.

    This is the primary entry point for this module. It dispatches to the
    correct language-specific extraction functions based on the file's language,
    then assembles the results into a FileEntities instance.

    WHY dispatch rather than unified logic:
        The node types in tree-sitter grammars are language-specific. A Python
        function is a "function_definition"; a JavaScript function is a
        "function_declaration" or "arrow_function" or "method_definition".
        There is no single traversal that correctly handles all four languages.
        Separate handlers per language also make it easy to add language-specific
        features (e.g. Python decorators, TypeScript interfaces) without
        tangling the logic.

    Args:
        parsed_file: A ParsedFile from tree_sitter_parser.parse_file().

    Returns:
        FileEntities containing all extracted functions, classes, and imports.
    """
    language = parsed_file.source_file.language
    relative_path = parsed_file.source_file.relative_path
    root = parsed_file.tree.root_node
    src = parsed_file.source_bytes

    logger.debug(f"Extracting entities from {relative_path} ({language})")

    extractor_map = {
        "python":     _extract_python_entities,
        "javascript": _extract_javascript_entities,
        "typescript": _extract_typescript_entities,
        "java":       _extract_java_entities,
        "go":         _extract_go_entities,
    }

    extractor = extractor_map.get(language)
    if extractor is None:
        logger.warning(f"No entity extractor for language '{language}', skipping.")
        return FileEntities(
            source_file_path=relative_path,
            language=language,
            functions=[],
            classes=[],
            imports=[],
        )

    return extractor(root, src, relative_path)


def extract_entities_batch(parsed_files: list[ParsedFile]) -> list[FileEntities]:
    """
    Extract entities from a list of ParsedFiles.

    WHY:
        Convenience batch wrapper. Handles per-file exceptions so a single
        malformed file doesn't abort extraction for the entire repo.

    Args:
        parsed_files: List of ParsedFile instances.

    Returns:
        List of FileEntities, one per successfully processed file.
    """
    results = []
    failed = 0
    for pf in parsed_files:
        try:
            entities = extract_entities(pf)
            results.append(entities)
        except Exception as exc:
            logger.error(
                f"Entity extraction failed for {pf.source_file.relative_path}: {exc}"
            )
            failed += 1
    logger.info(
        f"Entity extraction: {len(results)} succeeded, {failed} failed."
    )
    return results


# ---------------------------------------------------------------------------
# Python entity extraction
# ---------------------------------------------------------------------------

def _extract_python_entities(
    root: Node, src: bytes, relative_path: str
) -> FileEntities:
    """
    Extract functions, classes, and imports from a Python AST.

    Python-specific node types used:
        - function_definition       → def foo(...):
        - decorated_definition      → @decorator\ndef foo(...): (wraps function/class)
        - class_definition          → class Foo:
        - import_statement          → import os
        - import_from_statement     → from os import path
        - expression_statement      → contains string literals (used for docstrings)

    WHY we handle decorated_definition separately:
        In Python, a decorated function's node type is "decorated_definition",
        not "function_definition". The actual function definition is a child.
        We must unwrap the decorator to find the function name and signature.
    """
    functions: list[FunctionEntity] = []
    classes: list[ClassEntity] = []
    imports: list[ImportEntity] = []
    module_docstring: Optional[str] = _extract_python_module_docstring(root, src)

    # We walk the top-level children of the module node.
    # We recurse into classes separately to capture methods with their class context.
    for node in root.children:
        node_type = node.type

        if node_type == "function_definition":
            fn = _parse_python_function(node, src, relative_path, class_name=None)
            if fn:
                functions.append(fn)

        elif node_type == "decorated_definition":
            # Decorated functions/classes — unwrap to find the inner definition.
            inner = _get_decorated_inner(node)
            if inner and inner.type == "function_definition":
                decorators = _extract_python_decorators(node, src)
                fn = _parse_python_function(
                    inner, src, relative_path,
                    class_name=None, decorators=decorators
                )
                if fn:
                    functions.append(fn)
            elif inner and inner.type == "class_definition":
                decorators = _extract_python_decorators(node, src)
                cls = _parse_python_class(inner, src, relative_path, decorators, functions, classes)
                if cls:
                    classes.append(cls)

        elif node_type == "class_definition":
            cls = _parse_python_class(node, src, relative_path, [], functions, classes)
            if cls:
                classes.append(cls)

        elif node_type in ("import_statement", "import_from_statement"):
            imp = _parse_python_import(node, src, relative_path)
            if imp:
                imports.append(imp)

    return FileEntities(
        source_file_path=relative_path,
        language="python",
        functions=functions,
        classes=classes,
        imports=imports,
        module_docstring=module_docstring,
    )


def _parse_python_function(
    node: Node,
    src: bytes,
    relative_path: str,
    class_name: Optional[str],
    decorators: Optional[list[str]] = None,
) -> Optional[FunctionEntity]:
    """
    Build a FunctionEntity from a Python function_definition AST node.

    WHY we extract docstrings here:
        In Python, a docstring is the first expression statement in a function
        body, if that expression is a string literal. We detect and extract it
        here so it can be prepended to the embedding text — docstrings carry
        high-quality semantic content about intent.

    Args:
        node:         A tree-sitter "function_definition" node.
        src:          Raw source bytes of the file.
        relative_path: Relative file path for metadata.
        class_name:   Name of the containing class if this is a method.
        decorators:   Decorator names if the function was decorated.

    Returns:
        FunctionEntity, or None if the function name couldn't be resolved.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    params_node = node.child_by_field_name("parameters")
    parameters = _parse_python_parameters(params_node, src) if params_node else []

    return_type = None
    return_type_node = node.child_by_field_name("return_type")
    if return_type_node:
        # return_type node includes the "->" token; strip it for cleaner metadata.
        return_type = extract_node_text(return_type_node, src).lstrip("->").strip()

    body_node = node.child_by_field_name("body")
    body_text = extract_node_text(node, src)
    docstring = _extract_python_docstring_from_body(body_node, src) if body_node else None

    # Build a human-readable signature for display in query results.
    params_text = extract_node_text(params_node, src) if params_node else "()"
    ret_annotation = f" -> {return_type}" if return_type else ""
    signature = f"def {name}{params_text}{ret_annotation}:"

    is_async = any(c.type == "async" for c in node.children)
    start_line, end_line = get_node_line_range(node)

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        parameters=parameters,
        return_type=return_type,
        is_method=(class_name is not None),
        class_name=class_name,
        decorators=decorators or [],
        is_async=is_async,
    )


def _parse_python_class(
    node: Node,
    src: bytes,
    relative_path: str,
    decorators: list[str],
    functions_accumulator: list[FunctionEntity],
    classes_accumulator: list[ClassEntity],
) -> Optional[ClassEntity]:
    """
    Build a ClassEntity from a Python class_definition node and extract
    its methods into functions_accumulator.

    WHY we mutate functions_accumulator (and classes_accumulator):
        Methods are both part of the ClassEntity (as a name list) and
        standalone FunctionEntities in the functions list. Rather than
        returning a tuple, we write methods directly into the caller's
        function list and return just the ClassEntity. Nested classes
        (`class Outer: class Inner: ...`) work the same way — a class body
        is only visited by this function, never by the top-level
        _extract_python_entities loop, so without recursing into it here, a
        nested class (and everything inside it) would be silently invisible
        to indexing entirely, the same class of bug already fixed for Java's
        inner classes in this module. Nested classes are appended directly
        to classes_accumulator, not returned, since there's no "enclosing
        class" field on ClassEntity — they're indexed as their own entities.

    Args:
        node:                  "class_definition" tree-sitter node.
        src:                   Raw source bytes.
        relative_path:         Relative file path.
        decorators:            Class-level decorators.
        functions_accumulator: The function list to append extracted methods to.
        classes_accumulator:   The class list nested classes are appended to.

    Returns:
        ClassEntity, or None if the class name couldn't be resolved.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    class_name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    # Base classes from the argument list node.
    base_classes: list[str] = []
    superclasses_node = node.child_by_field_name("superclasses")
    if superclasses_node:
        for child in superclasses_node.children:
            if child.type not in (",", "(", ")"):
                base_classes.append(extract_node_text(child, src).strip())

    # Extract class docstring.
    body_node = node.child_by_field_name("body")
    docstring = _extract_python_docstring_from_body(body_node, src) if body_node else None

    # Extract methods from the class body.
    method_names: list[str] = []
    if body_node:
        for child in body_node.children:
            if child.type == "function_definition":
                method = _parse_python_function(child, src, relative_path, class_name)
                if method:
                    method_names.append(method.name)
                    functions_accumulator.append(method)
            elif child.type == "decorated_definition":
                inner = _get_decorated_inner(child)
                if inner and inner.type == "function_definition":
                    decs = _extract_python_decorators(child, src)
                    method = _parse_python_function(
                        inner, src, relative_path, class_name, decs
                    )
                    if method:
                        method_names.append(method.name)
                        functions_accumulator.append(method)
                elif inner and inner.type == "class_definition":
                    decs = _extract_python_decorators(child, src)
                    nested = _parse_python_class(
                        inner, src, relative_path, decs, functions_accumulator, classes_accumulator
                    )
                    if nested:
                        classes_accumulator.append(nested)
            elif child.type == "class_definition":
                nested = _parse_python_class(
                    child, src, relative_path, [], functions_accumulator, classes_accumulator
                )
                if nested:
                    classes_accumulator.append(nested)

    return ClassEntity(
        name=class_name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=method_names,
        docstring=docstring,
        decorators=decorators,
    )


def _parse_python_import(
    node: Node, src: bytes, relative_path: str
) -> Optional[ImportEntity]:
    """
    Parse a Python import_statement or import_from_statement node.

    Handles all Python import forms:
        import os                                → module_path="os"
        import os.path                           → module_path="os.path"
        import numpy as np                       → module_path="numpy", alias="np"
        from os import path                      → module_path="os", names=["path"]
        from . import utils                      → is_relative=True
        from ..auth import login, logout         → is_relative=True, names=[...]

    Args:
        node:          "import_statement" or "import_from_statement" node.
        src:           Raw source bytes.
        relative_path: Relative file path.

    Returns:
        ImportEntity, or None if the module path cannot be extracted.
    """
    raw = extract_node_text(node, src)
    start_line, _ = get_node_line_range(node)

    if node.type == "import_statement":
        # "import X" or "import X as Y"
        name_nodes = [c for c in node.children if c.type in ("dotted_name", "aliased_import")]
        if not name_nodes:
            return None
        first = name_nodes[0]
        if first.type == "aliased_import":
            name_part = first.child_by_field_name("name")
            alias_part = first.child_by_field_name("alias")
            module_path = extract_node_text(name_part, src) if name_part else ""
            alias = extract_node_text(alias_part, src) if alias_part else None
        else:
            module_path = extract_node_text(first, src)
            alias = None
        return ImportEntity(
            raw_statement=raw,
            module_path=module_path,
            relative_file_path=relative_path,
            language="python",
            start_line=start_line,
            alias=alias,
        )

    elif node.type == "import_from_statement":
        # "from X import Y" — relative imports have "." or ".." as module.
        module_node = node.child_by_field_name("module_name")
        module_path = extract_node_text(module_node, src) if module_node else ""
        is_relative = module_path.startswith(".")

        imported_names: list[str] = []
        for child in node.children:
            if child.type == "import_prefix":
                continue  # the "." or ".." prefix
            if child.type in ("dotted_name", "identifier"):
                if child != module_node:
                    imported_names.append(extract_node_text(child, src))
            elif child.type == "import_list":
                for name_child in child.children:
                    if name_child.type in ("identifier", "dotted_name"):
                        imported_names.append(extract_node_text(name_child, src))

        return ImportEntity(
            raw_statement=raw,
            module_path=module_path,
            relative_file_path=relative_path,
            language="python",
            start_line=start_line,
            imported_names=imported_names,
            is_relative=is_relative,
        )

    return None


# ---------------------------------------------------------------------------
# JavaScript / TypeScript entity extraction
# ---------------------------------------------------------------------------

def _extract_javascript_entities(
    root: Node, src: bytes, relative_path: str
) -> FileEntities:
    """
    Extract entities from a JavaScript AST.

    JavaScript is complex because functions can be defined in multiple ways:
        - function foo() {}         → function_declaration
        - const foo = function() {} → variable_declarator + function
        - const foo = () => {}      → variable_declarator + arrow_function
        - class Foo { bar() {} }    → class_declaration + method_definition

    We handle the most common patterns. Arrow functions assigned to variables
    are treated as named functions using the variable name.
    """
    functions: list[FunctionEntity] = []
    classes: list[ClassEntity] = []
    imports: list[ImportEntity] = []

    for node in walk_tree(root):
        if node.type == "function_declaration":
            fn = _parse_js_function_declaration(node, src, relative_path)
            if fn:
                functions.append(fn)

        elif node.type in ("class_declaration", "abstract_class_declaration"):
            # abstract_class_declaration (TypeScript `abstract class Foo {}`)
            # is a distinct top-level node type from class_declaration, not
            # a modifier on it — without this branch, every abstract class
            # was invisible to indexing entirely. It shares the same
            # name/body field layout, so _parse_js_class handles both.
            cls = _parse_js_class(node, src, relative_path, functions)
            if cls:
                classes.append(cls)

        elif node.type == "import_statement":
            imp = _parse_js_import(node, src, relative_path)
            if imp:
                imports.append(imp)

        elif node.type == "variable_declarator":
            fn = _parse_js_variable_function(node, src, relative_path)
            if fn:
                functions.append(fn)

    return FileEntities(
        source_file_path=relative_path,
        language="javascript",
        functions=functions,
        classes=classes,
        imports=imports,
    )


def _extract_typescript_entities(
    root: Node, src: bytes, relative_path: str
) -> FileEntities:
    """
    Extract entities from a TypeScript AST.

    TypeScript is a superset of JavaScript, so most JS extraction logic applies.
    Additional TypeScript constructs handled here:
        - interface_declaration (treated as ClassEntity with is_interface=True)
        - type annotations on function parameters and return types
    """
    # TypeScript shares most node types with JavaScript.
    entities = _extract_javascript_entities(root, src, relative_path)
    entities.language = "typescript"

    # Additionally extract TypeScript interfaces.
    for node in walk_tree(root):
        if node.type == "interface_declaration":
            iface = _parse_ts_interface(node, src, relative_path)
            if iface:
                entities.classes.append(iface)

    return entities


def _parse_js_function_declaration(
    node: Node, src: bytes, relative_path: str,
    class_name: Optional[str] = None,
) -> Optional[FunctionEntity]:
    """
    Parse a JavaScript/TypeScript function_declaration or method_definition.

    Args:
        node:         "function_declaration" or "method_definition" node.
        src:          Raw source bytes.
        relative_path: Relative file path.
        class_name:   Enclosing class name, if this is a method.

    Returns:
        FunctionEntity, or None if name cannot be resolved.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    params_node = node.child_by_field_name("parameters")
    parameters: list[str] = []
    if params_node:
        for child in params_node.children:
            if child.type not in (",", "(", ")", "required_parameter", "optional_parameter"):
                text = extract_node_text(child, src).strip()
                if text:
                    parameters.append(text)

    body_text = extract_node_text(node, src)
    params_text = extract_node_text(params_node, src) if params_node else "()"
    is_async = any(c.type == "async" for c in node.children)
    signature = f"{'async ' if is_async else ''}function {name}{params_text}"
    start_line, end_line = get_node_line_range(node)

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="javascript",
        start_line=start_line,
        end_line=end_line,
        parameters=parameters,
        is_method=(class_name is not None),
        class_name=class_name,
        is_async=is_async,
    )


def _parse_js_variable_function(
    node: Node, src: bytes, relative_path: str
) -> Optional[FunctionEntity]:
    """
    Parse a `const Foo = () => {...}` / `const Foo = function() {...}` /
    `const Foo = forwardRef((props, ref) => {...})` variable_declarator into
    a FunctionEntity.

    WHY THIS EXISTS:
        Modern JS/React code overwhelmingly declares functions, hooks, and
        components as `const` bindings rather than `function` declarations
        — including higher-order-component wrappers like React.forwardRef
        and React.memo (`export const Button = forwardRef((props, ref) =>
        ...)`). Without this, a whole file of such declarations (a typical
        shared UI-primitives file, for instance) would produce zero indexed
        entities even though `function_declaration` handling exists —
        despite this module's own docstring claiming to support exactly
        this pattern ("const foo = () => {}").

        Plain non-function declarators (`const x = 5`, `const { a } = obj`)
        are skipped by checking the declared value's node type.
    """
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    if not name_node or value_node is None or name_node.type != "identifier":
        return None  # skip destructuring patterns and declarations with no initializer

    fn_node = value_node
    if value_node.type == "call_expression":
        # Higher-order-component pattern: forwardRef(fn), memo(fn), etc. —
        # use the first function-like argument, if any.
        args_node = value_node.child_by_field_name("arguments")
        fn_node = None
        if args_node:
            for arg in args_node.children:
                if arg.type in ("arrow_function", "function_expression"):
                    fn_node = arg
                    break
        if fn_node is None:
            return None

    if fn_node.type not in ("arrow_function", "function_expression"):
        return None

    name = extract_node_text(name_node, src)
    # Use the full declarator's text (`Foo = forwardRef(...)`) as the body,
    # not just the inner function — the HOC wrapper is meaningful context.
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    params_node = (
        fn_node.child_by_field_name("parameters")
        or fn_node.child_by_field_name("parameter")
    )
    params_text = extract_node_text(params_node, src) if params_node else "()"
    is_async = any(c.type == "async" for c in fn_node.children)

    signature = f"{'async ' if is_async else ''}const {name} = {params_text} => ..."

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="javascript",
        start_line=start_line,
        end_line=end_line,
        parameters=[],
        is_method=False,
        class_name=None,
        is_async=is_async,
    )


def _collect_heritage_names(clause_node: Node, src: bytes) -> list[str]:
    """
    Collect base type names from one extends_clause/implements_clause
    (class_heritage) or extends_type_clause (interface) node.

    WHY ITERATE DIRECT CHILDREN INSTEAD OF A FULL RECURSIVE WALK:
        A qualified base name like `extends React.Component` or
        `implements B.C` parses as ONE member_expression/nested_type_identifier
        node containing "React"/"Component" (or "B"/"C") as separate child
        identifiers. Recursively walking every descendant and matching bare
        "identifier"/"type_identifier" leaves — the previous approach — would
        record "React" alone, silently dropping ".Component" and producing a
        base class name that doesn't match anything. Only direct children of
        the clause are inspected, and each qualified-name node's full text is
        taken as one unit.

        A generic base (`extends Base<T>`) is a "generic_type" node — its
        type_arguments ("<T>") are stripped by using its "type" field rather
        than the whole node's text, so the recorded name is "Base", matching
        how the type itself is actually named elsewhere in the file.
    """
    names: list[str] = []
    for child in clause_node.children:
        if child.type in (
            "identifier", "type_identifier", "nested_type_identifier", "member_expression",
        ):
            names.append(extract_node_text(child, src))
        elif child.type == "generic_type":
            # NOTE: TypeScript's generic_type names this field "name", not
            # "type" — different from Go's generic_type (see
            # _resolve_go_receiver_type), which really does use "type".
            # Different grammars, same node type name, different field name.
            name_field = child.child_by_field_name("name")
            if name_field:
                names.append(extract_node_text(name_field, src))
    return names


def _parse_js_class(
    node: Node,
    src: bytes,
    relative_path: str,
    functions_accumulator: list[FunctionEntity],
) -> Optional[ClassEntity]:
    """
    Parse a JavaScript/TypeScript class_declaration node.

    Args:
        node:                  "class_declaration" node.
        src:                   Raw source bytes.
        relative_path:         Relative file path.
        functions_accumulator: Function list to append methods to.

    Returns:
        ClassEntity, or None if name is missing.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    class_name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    # Heritage clause: "extends Foo implements Bar, Baz". class_heritage is
    # an UNNAMED child (not accessible via child_by_field_name("heritage") —
    # that field name doesn't exist in this grammar and always returned
    # None, silently leaving base_classes empty for every JS/TS class ever
    # parsed by this function, extends/implements included).
    base_classes: list[str] = []
    for heritage in find_nodes_by_type(node, "class_heritage"):
        for clause in heritage.children:
            if clause.type in ("extends_clause", "implements_clause"):
                base_classes.extend(_collect_heritage_names(clause, src))

    # Extract methods from class body. abstract_method_signature (TS
    # `abstract foo(): void;`, no body) is included alongside
    # method_definition since this function is also used to parse
    # abstract_class_declaration, whose body mixes both.
    method_names: list[str] = []
    body_node = node.child_by_field_name("body")
    if body_node:
        for child in body_node.children:
            if child.type in ("method_definition", "abstract_method_signature"):
                fn = _parse_js_function_declaration(child, src, relative_path, class_name)
                if fn:
                    method_names.append(fn.name)
                    functions_accumulator.append(fn)

    return ClassEntity(
        name=class_name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="javascript",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=method_names,
    )


def _parse_js_import(
    node: Node, src: bytes, relative_path: str
) -> Optional[ImportEntity]:
    """
    Parse a JavaScript/TypeScript import_statement node.

    Handles:
        import React from 'react'
        import { useState, useEffect } from 'react'
        import * as ReactDOM from 'react-dom'
        import type { Foo } from './types'   (TypeScript)

    Args:
        node:          "import_statement" node.
        src:           Raw source bytes.
        relative_path: Relative file path.

    Returns:
        ImportEntity, or None if module path is missing.
    """
    raw = extract_node_text(node, src)
    start_line, _ = get_node_line_range(node)

    source_node = node.child_by_field_name("source")
    if not source_node:
        return None

    # Strip surrounding quotes from the module path string literal.
    module_path = extract_node_text(source_node, src).strip("'\"")
    is_relative = module_path.startswith(".")

    imported_names: list[str] = []
    for child in node.children:
        if child.type == "import_clause":
            for sub in walk_tree(child):
                if sub.type == "identifier":
                    imported_names.append(extract_node_text(sub, src))

    return ImportEntity(
        raw_statement=raw,
        module_path=module_path,
        relative_file_path=relative_path,
        language="javascript",
        start_line=start_line,
        imported_names=imported_names,
        is_relative=is_relative,
    )


def _parse_ts_interface(
    node: Node, src: bytes, relative_path: str
) -> Optional[ClassEntity]:
    """
    Parse a TypeScript interface_declaration node as a ClassEntity.

    WHY treat interfaces like classes:
        For the purposes of CodeSense's call graph and semantic search,
        interfaces function like abstract classes — they define method
        signatures that concrete classes implement. Storing them as ClassEntity
        with is_interface=True lets us answer "what interfaces does FooService
        implement?" using the same query path as class inheritance.

    Args:
        node:          "interface_declaration" node.
        src:           Raw source bytes.
        relative_path: Relative file path.

    Returns:
        ClassEntity with is_interface=True, or None if name is missing.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    # extends_type_clause ("interface Shape extends Base, Other {}") is an
    # unnamed child, same reason class_heritage needed find_nodes_by_type
    # in _parse_js_class rather than a field-name lookup.
    base_classes: list[str] = []
    for clause in find_nodes_by_type(node, "extends_type_clause"):
        base_classes.extend(_collect_heritage_names(clause, src))

    # Interface method signatures ("area(): number;") have no body, so they
    # use "method_signature" — a distinct node type from method_definition
    # (which requires a body) — this was never checked, so every interface
    # reported an empty `methods` list regardless of what it declared.
    methods: list[str] = []
    body_node = node.child_by_field_name("body")
    if body_node:
        for sig in find_nodes_by_type(body_node, "method_signature"):
            name_field = sig.child_by_field_name("name")
            if name_field:
                methods.append(extract_node_text(name_field, src))

    return ClassEntity(
        name=name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="typescript",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=methods,
        is_interface=True,
    )


# ---------------------------------------------------------------------------
# Java entity extraction
# ---------------------------------------------------------------------------

def _extract_java_entities(
    root: Node, src: bytes, relative_path: str
) -> FileEntities:
    """
    Extract entities from a Java AST.

    Java-specific node types:
        - class_declaration         → public class Foo { }
        - interface_declaration     → public interface Bar { }
        - enum_declaration          → public enum Status { ACTIVE, INACTIVE }
        - record_declaration        → public record Point(int x, int y) { }
        - method_declaration        → public void doThing() { }
        - constructor_declaration   → public Foo() { }
        - import_declaration        → import java.util.List;

    WHY we handle constructors as functions:
        Constructors are callable units that appear in call graphs. Treating
        them as FunctionEntities lets impact analysis trace "what breaks if
        the constructor of X changes?" consistently with regular methods.

    WHY NESTED TYPES ARE RECURSED INTO:
        A class/enum/record's body can itself contain nested class/interface/
        enum/record declarations (inner classes, the Builder pattern, a
        nested enum for a field's valid values, etc.) — these live inside
        the outer type's `body`/`class_body` node, never at the top level of
        `root.children`, so a top-level-only loop like this one's would never
        see them. _parse_java_type_member recurses into each type's body for
        exactly this reason, appending nested types to the same shared
        `classes` list as top-level ones (there's no "enclosing class" field
        on ClassEntity — nested types are indexed as their own entities,
        consistent with how this module already treats every language).
    """
    functions: list[FunctionEntity] = []
    classes: list[ClassEntity] = []
    imports: list[ImportEntity] = []

    for node in root.children:
        if node.type == "import_declaration":
            imp = _parse_java_import(node, src, relative_path)
            if imp:
                imports.append(imp)
        else:
            _parse_java_type_member(node, src, relative_path, functions, classes)

    return FileEntities(
        source_file_path=relative_path,
        language="java",
        functions=functions,
        classes=classes,
        imports=imports,
    )


def _parse_java_type_member(
    node: Node,
    src: bytes,
    relative_path: str,
    functions_accumulator: list[FunctionEntity],
    classes_accumulator: list[ClassEntity],
) -> None:
    """
    Dispatch a single type-declaration node (class/interface/enum/record) to
    its parser and append the result to classes_accumulator. Node types this
    doesn't recognize (field_declaration, static/instance initializer
    blocks, etc.) are silently ignored, same as before this function existed.

    Shared by both the top-level entity-extraction loop and each type's own
    body — nested types are dispatched through this same function, which is
    what makes inner classes/enums/records visible at all (see
    _extract_java_entities' docstring for why that matters).
    """
    if node.type in ("class_declaration", "interface_declaration"):
        cls = _parse_java_class(node, src, relative_path, functions_accumulator, classes_accumulator)
        if cls:
            classes_accumulator.append(cls)
    elif node.type == "enum_declaration":
        cls = _parse_java_enum(node, src, relative_path, functions_accumulator, classes_accumulator)
        if cls:
            classes_accumulator.append(cls)
    elif node.type == "record_declaration":
        cls = _parse_java_record(node, src, relative_path, functions_accumulator, classes_accumulator)
        if cls:
            classes_accumulator.append(cls)


def _parse_java_class(
    node: Node,
    src: bytes,
    relative_path: str,
    functions_accumulator: list[FunctionEntity],
    classes_accumulator: list[ClassEntity],
) -> Optional[ClassEntity]:
    """
    Parse a Java class_declaration or interface_declaration.

    Args:
        node:                  "class_declaration" or "interface_declaration" node.
        src:                   Raw source bytes.
        relative_path:         Relative file path.
        functions_accumulator: Function list to append methods to.
        classes_accumulator:   Class list nested types are appended to
                                directly (see _extract_java_entities'
                                docstring for why nested types need this).

    Returns:
        ClassEntity, or None if the class name is missing.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    class_name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)
    is_interface = node.type == "interface_declaration"

    # Superclass and interfaces.
    base_classes: list[str] = []
    superclass_node = node.child_by_field_name("superclass")
    if superclass_node:
        base_classes.append(extract_node_text(superclass_node, src))
    interfaces_node = node.child_by_field_name("interfaces")
    if interfaces_node:
        for child in walk_tree(interfaces_node):
            if child.type == "type_identifier":
                base_classes.append(extract_node_text(child, src))

    # Extract methods, constructors, and nested types.
    method_names: list[str] = []
    body_node = node.child_by_field_name("body")
    if body_node:
        for child in body_node.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                fn = _parse_java_method(child, src, relative_path, class_name)
                if fn:
                    method_names.append(fn.name)
                    functions_accumulator.append(fn)
            else:
                _parse_java_type_member(
                    child, src, relative_path, functions_accumulator, classes_accumulator
                )

    return ClassEntity(
        name=class_name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="java",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=method_names,
        docstring=_extract_java_doc_comment(node, src),
        is_interface=is_interface,
    )


def _parse_java_enum(
    node: Node,
    src: bytes,
    relative_path: str,
    functions_accumulator: list[FunctionEntity],
    classes_accumulator: list[ClassEntity],
) -> Optional[ClassEntity]:
    """
    Parse a Java enum_declaration.

    Enums are structurally close to classes — they can implement interfaces,
    declare fields, a constructor, and regular methods — the only difference
    relevant here is that the body is "enum_body" (constants, then an
    optional "enum_body_declarations" holding everything else) rather than
    "class_body". Constant-specific method bodies (an enum constant like
    `ACTIVE("A") { public String extra() {...} }` overriding a method) are
    intentionally not descended into — that's a rare pattern and the
    constant's body is still captured verbatim in the enum's own body_text.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    enum_name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    base_classes: list[str] = []
    interfaces_node = node.child_by_field_name("interfaces")
    if interfaces_node:
        for child in walk_tree(interfaces_node):
            if child.type == "type_identifier":
                base_classes.append(extract_node_text(child, src))

    method_names: list[str] = []
    body_node = node.child_by_field_name("body")
    declarations = find_nodes_by_type(body_node, "enum_body_declarations") if body_node else []
    for decls in declarations:
        for child in decls.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                fn = _parse_java_method(child, src, relative_path, enum_name)
                if fn:
                    method_names.append(fn.name)
                    functions_accumulator.append(fn)
            else:
                _parse_java_type_member(
                    child, src, relative_path, functions_accumulator, classes_accumulator
                )

    return ClassEntity(
        name=enum_name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="java",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=method_names,
        docstring=_extract_java_doc_comment(node, src),
        is_interface=False,
    )


def _parse_java_record(
    node: Node,
    src: bytes,
    relative_path: str,
    functions_accumulator: list[FunctionEntity],
    classes_accumulator: list[ClassEntity],
) -> Optional[ClassEntity]:
    """
    Parse a Java record_declaration (Java 14+), e.g.
    `public record Point(int x, int y) implements Shape { ... }`.

    A record's components (`(int x, int y)`) implicitly generate accessor
    methods, a constructor, equals/hashCode/toString — none of which have
    source bodies to index, so they're deliberately not synthesized as
    FunctionEntity objects here (there is no line range or body_text for
    generated code). Only explicitly-written methods in the record's body
    are extracted, same as a class.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    record_name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    base_classes: list[str] = []
    interfaces_node = node.child_by_field_name("interfaces")
    if interfaces_node:
        for child in walk_tree(interfaces_node):
            if child.type == "type_identifier":
                base_classes.append(extract_node_text(child, src))

    method_names: list[str] = []
    body_node = node.child_by_field_name("body")
    if body_node:
        for child in body_node.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                fn = _parse_java_method(child, src, relative_path, record_name)
                if fn:
                    method_names.append(fn.name)
                    functions_accumulator.append(fn)
            else:
                _parse_java_type_member(
                    child, src, relative_path, functions_accumulator, classes_accumulator
                )

    return ClassEntity(
        name=record_name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="java",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=method_names,
        docstring=_extract_java_doc_comment(node, src),
        is_interface=False,
    )


def _extract_java_doc_comment(node: Node, src: bytes) -> Optional[str]:
    """
    Extract a Java doc comment: the Javadoc block comment ("/** ... */") or
    contiguous line-comment block immediately preceding `node`, with no
    blank line in between.

    WHY THIS MATTERS BEYOND DOCUMENTATION:
        Prior to this, NEITHER Java classes NOR methods ever populated
        `docstring` — despite `/** ... */` Javadoc being the idiomatic,
        widely-used convention for exactly this purpose in Java (annotations
        like @Controller sit inside the declaration's own `modifiers` node,
        not between the comment and the declaration, so they don't block
        this lookup). CodeChunk.embeddable_text (indexing/chunk_schema.py)
        already treats `docstring` as the highest-signal natural-language
        text for embedding, the same mechanism already relied on for
        Python docstrings and (as of this session) Go doc comments — Java
        was silently getting none of that benefit, which matters more as
        the corpus grows since bare method signatures rarely paraphrase
        well against an English query on their own.
    """
    current = node.prev_sibling
    if current is None or current.type not in ("line_comment", "block_comment"):
        return None
    if current.end_point[0] != node.start_point[0] - 1:
        return None

    if current.type == "block_comment":
        text = extract_node_text(current, src).strip()
        if text.startswith("/**"):
            text = text[3:]
        elif text.startswith("/*"):
            text = text[2:]
        if text.endswith("*/"):
            text = text[:-2]
        cleaned = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("*"):
                line = line[1:].strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned) if cleaned else None

    # line_comment: walk backward collecting a contiguous `//` block, same
    # adjacency rule as the block_comment case above.
    lines: list[str] = []
    expected_end_row = node.start_point[0] - 1
    while current is not None and current.type == "line_comment":
        if current.end_point[0] != expected_end_row:
            break
        text = extract_node_text(current, src).strip()
        if text.startswith("//"):
            text = text[2:].strip()
        lines.append(text)
        expected_end_row = current.start_point[0] - 1
        current = current.prev_sibling
    lines.reverse()
    return "\n".join(lines) if lines else None


def _parse_java_method(
    node: Node, src: bytes, relative_path: str, class_name: str
) -> Optional[FunctionEntity]:
    """
    Parse a Java method_declaration or constructor_declaration.

    Args:
        node:          "method_declaration" or "constructor_declaration" node.
        src:           Raw source bytes.
        relative_path: Relative file path.
        class_name:    Enclosing class name.

    Returns:
        FunctionEntity representing the method, or None if name is missing.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    params_node = node.child_by_field_name("parameters")
    params_text = extract_node_text(params_node, src) if params_node else "()"

    return_type_node = node.child_by_field_name("type")
    return_type = extract_node_text(return_type_node, src) if return_type_node else None
    ret_annotation = f" {return_type}" if return_type else ""

    signature = f"{ret_annotation} {class_name}.{name}{params_text}".strip()
    docstring = _extract_java_doc_comment(node, src)

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="java",
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        return_type=return_type,
        is_method=True,
        class_name=class_name,
    )


def _parse_java_import(
    node: Node, src: bytes, relative_path: str
) -> Optional[ImportEntity]:
    """
    Parse a Java import_declaration node.

    Handles:
        import java.util.List;
        import java.util.*;    (wildcard)
        import static org.junit.Assert.assertEquals;

    Args:
        node:          "import_declaration" node.
        src:           Raw source bytes.
        relative_path: Relative file path.

    Returns:
        ImportEntity, or None if the import path is empty.
    """
    raw = extract_node_text(node, src)
    start_line, _ = get_node_line_range(node)

    # The full import path is the text after "import" and optional "static".
    # Strip the semicolon from the raw text to get a clean module path.
    module_path = raw.replace("import", "").replace("static", "").strip().rstrip(";").strip()
    if not module_path:
        return None

    # For "import java.util.List", the imported name is "List".
    parts = module_path.split(".")
    imported_names = [parts[-1]] if parts[-1] != "*" else []

    return ImportEntity(
        raw_statement=raw,
        module_path=module_path,
        relative_file_path=relative_path,
        language="java",
        start_line=start_line,
        imported_names=imported_names,
    )


# ---------------------------------------------------------------------------
# Go entity extraction
# ---------------------------------------------------------------------------

def _extract_go_entities(
    root: Node, src: bytes, relative_path: str
) -> FileEntities:
    """
    Extract entities from a Go AST.

    Go-specific node types:
        - function_declaration  → func Foo(...) { }
        - method_declaration    → func (s *Store) Get(...) { }
        - type_declaration      → type Foo struct { } / type Bar interface { }
                                   (single or grouped: `type ( A struct{}; B
                                   interface{} )` nests multiple type_spec
                                   children directly, no wrapper node)
        - import_declaration    → import "fmt" / import ( "fmt"; r "net/http" )

    WHY METHODS ARE LINKED IN A SEPARATE STEP:
        Unlike Python/JS/Java, Go methods are not nested inside the struct's
        body — they are top-level declarations elsewhere in the file that
        reference their type via a receiver parameter, e.g.
        `func (s *Store) Get(...)`. So struct/interface types are collected
        first, then each method is matched back to its receiver's type name
        (pointer stripped) to populate ClassEntity.methods, mirroring what
        Java/Python get natively from AST nesting.
    """
    functions: list[FunctionEntity] = []
    classes: list[ClassEntity] = []
    imports: list[ImportEntity] = []
    classes_by_name: dict[str, ClassEntity] = {}

    # First pass: imports and type declarations, so every struct/interface
    # is known before we try to link methods to their receiver type below.
    for node in root.children:
        if node.type == "import_declaration":
            imports.extend(_parse_go_imports(node, src, relative_path))
        elif node.type == "type_declaration":
            for cls in _parse_go_type_declaration(node, src, relative_path):
                classes.append(cls)
                classes_by_name[cls.name] = cls

    # Second pass: top-level functions and receiver methods.
    for node in root.children:
        if node.type == "function_declaration":
            fn = _parse_go_function(node, src, relative_path)
            if fn:
                functions.append(fn)
        elif node.type == "method_declaration":
            fn = _parse_go_method(node, src, relative_path)
            if fn:
                functions.append(fn)
                cls = classes_by_name.get(fn.class_name)
                if cls:
                    cls.methods.append(fn.name)

    return FileEntities(
        source_file_path=relative_path,
        language="go",
        functions=functions,
        classes=classes,
        imports=imports,
    )


def _extract_go_doc_comment(node: Node, src: bytes) -> Optional[str]:
    """
    Extract a Go doc comment: the contiguous block of comments immediately
    preceding `node`, with no blank line in between.

    WHY THIS MATTERS BEYOND DOCUMENTATION:
        Go has no docstring syntax — by convention (godoc), the natural-
        language description of a function or type is the `//` comment
        block directly above it, e.g.:
            // CreateSession writes the session hash and sets the absolute TTL.
            func (s *Store) CreateSession(...) error { ... }
        This is the ONLY natural-language text available for a Go entity, and
        FileEntities' consumers (indexing/chunk_schema.CodeChunk.embeddable_text)
        already prioritize `docstring` as high-signal text for embedding —
        exactly like Python docstrings or Java Javadoc. Without extracting it,
        terse Go code (identifiers and syntax only) embeds poorly against an
        English query, regardless of which embedding model is used — the doc
        comment is what bridges "stores the session with a TTL" to a query
        like "where is data stored temporarily".

    Comments are separate sibling nodes in this grammar (not attached to the
    declaration as a field), so we walk backward through `prev_sibling` while
    each node is a comment on the line directly above the previous one —
    stopping at the first blank line or non-comment node, which is what
    godoc itself does to decide where a doc comment block starts.
    """
    lines: list[str] = []
    current = node.prev_sibling
    expected_end_row = node.start_point[0] - 1
    while current is not None and current.type == "comment":
        if current.end_point[0] != expected_end_row:
            break
        text = extract_node_text(current, src).strip()
        if text.startswith("//"):
            text = text[2:].strip()
        elif text.startswith("/*") and text.endswith("*/"):
            text = text[2:-2].strip()
        lines.append(text)
        expected_end_row = current.start_point[0] - 1
        current = current.prev_sibling
    lines.reverse()
    return "\n".join(lines) if lines else None


def _parse_go_function(
    node: Node, src: bytes, relative_path: str
) -> Optional[FunctionEntity]:
    """Parse a Go function_declaration (no receiver)."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    params_node = node.child_by_field_name("parameters")
    parameters = _parse_go_parameters(params_node, src) if params_node else []
    params_text = extract_node_text(params_node, src) if params_node else "()"

    result_node = node.child_by_field_name("result")
    return_type = extract_node_text(result_node, src) if result_node else None
    ret_annotation = f" {return_type}" if return_type else ""

    signature = f"func {name}{params_text}{ret_annotation}".strip()
    docstring = _extract_go_doc_comment(node, src)

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="go",
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        parameters=parameters,
        return_type=return_type,
        is_method=False,
        class_name=None,
    )


def _parse_go_method(
    node: Node, src: bytes, relative_path: str
) -> Optional[FunctionEntity]:
    """
    Parse a Go method_declaration — a function_declaration plus a receiver
    parameter, e.g. `func (s *Store) Get(key string) (string, bool) { }`.
    """
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)

    receiver_node = node.child_by_field_name("receiver")
    class_name = _resolve_go_receiver_type(receiver_node, src) if receiver_node else None
    receiver_text = extract_node_text(receiver_node, src) if receiver_node else ""

    params_node = node.child_by_field_name("parameters")
    parameters = _parse_go_parameters(params_node, src) if params_node else []
    params_text = extract_node_text(params_node, src) if params_node else "()"

    result_node = node.child_by_field_name("result")
    return_type = extract_node_text(result_node, src) if result_node else None
    ret_annotation = f" {return_type}" if return_type else ""

    # receiver_text is already parenthesized (it's a parameter_list's own
    # text, e.g. "(s *Store)") — don't wrap it in another pair of parens.
    signature = f"func {receiver_text} {name}{params_text}{ret_annotation}".strip()
    docstring = _extract_go_doc_comment(node, src)

    return FunctionEntity(
        name=name,
        signature=signature,
        body_text=body_text,
        relative_file_path=relative_path,
        language="go",
        start_line=start_line,
        end_line=end_line,
        docstring=docstring,
        parameters=parameters,
        return_type=return_type,
        is_method=True,
        class_name=class_name,
    )


def _resolve_go_receiver_type(receiver_node: Node, src: bytes) -> Optional[str]:
    """
    Extract the receiver's struct/interface type name from a method's
    receiver parameter list.

    "(s *Store)"     → "Store" (pointer receiver)
    "(s Store)"      → "Store" (value receiver)
    "(s *Stack[T])"  → "Stack" (generic receiver — type arguments stripped)

    The pointer marker is stripped because a value receiver and a pointer
    receiver on the same type both belong to the same ClassEntity for our
    purposes (linking the method into that type's `methods` list). Generic
    type arguments are stripped for the same reason: a generic struct's
    type_spec is recorded under its bare name ("Stack"), not "Stack[T]" —
    without stripping this, every method of a generic type would fail to
    link to its struct (exact string match against classes_by_name), which
    is silent (no crash — the method just orphans from its class' method
    list) and easy to miss without a generic-type test case.
    """
    for param in find_nodes_by_type(receiver_node, "parameter_declaration"):
        type_node = param.child_by_field_name("type")
        if type_node is None:
            continue
        if type_node.type == "pointer_type":
            # pointer_type's inner type_identifier has no field name — it's
            # simply the last child after the "*" token.
            type_node = type_node.children[-1]
        if type_node.type == "generic_type":
            type_node = type_node.child_by_field_name("type")
            if type_node is None:
                continue
        return extract_node_text(type_node, src)
    return None


def _parse_go_parameters(params_node: Node, src: bytes) -> list[str]:
    """Return each parameter_declaration's source text as a string."""
    return [
        extract_node_text(p, src)
        for p in iter_children_of_type(params_node, "parameter_declaration")
    ]


def _parse_go_type_declaration(
    node: Node, src: bytes, relative_path: str
) -> list[ClassEntity]:
    """
    Parse a Go type_declaration into zero or more ClassEntity instances.

    Handles both a single declaration (`type Foo struct { }`, one type_spec
    child) and a grouped block (`type ( A struct{}; B interface{} )`,
    multiple type_spec children) — both forms nest type_spec directly under
    type_declaration in this grammar, with no wrapper node to distinguish.
    """
    specs = [c for c in node.children if c.type == "type_spec"]
    classes: list[ClassEntity] = []
    for spec in specs:
        cls = _parse_go_type_spec(spec, src, relative_path)
        if cls:
            if cls.docstring is None and len(specs) == 1:
                # A single (non-grouped) `type Foo struct { }` declaration's
                # doc comment precedes the whole type_declaration node, not
                # the inner type_spec — only grouped `type ( ... )` blocks
                # attach comments directly above each type_spec.
                cls.docstring = _extract_go_doc_comment(node, src)
            classes.append(cls)
    return classes


def _parse_go_type_spec(
    node: Node, src: bytes, relative_path: str
) -> Optional[ClassEntity]:
    """
    Parse a single Go type_spec, if it defines a struct or interface.

    Other type_spec forms (`type ID string`, `type Handler func(...) error`,
    type aliases) are plain type definitions with no methods or fields worth
    indexing as their own entity, so they are skipped (return None).
    """
    name_node = node.child_by_field_name("name")
    type_node = node.child_by_field_name("type")
    if not name_node or not type_node:
        return None
    if type_node.type not in ("struct_type", "interface_type"):
        return None

    name = extract_node_text(name_node, src)
    body_text = extract_node_text(node, src)
    start_line, end_line = get_node_line_range(node)
    is_interface = type_node.type == "interface_type"

    # Interface methods are declared inline in the interface body (there is
    # no separate method_declaration for these, unlike struct methods which
    # live at package level with a receiver) — record them directly.
    methods: list[str] = []
    base_classes: list[str] = []
    if is_interface:
        for elem in find_nodes_by_type(type_node, "method_elem"):
            name_field = elem.child_by_field_name("name")
            if name_field:
                methods.append(extract_node_text(name_field, src))
        # Interface embedding (`type BindingBody interface { Binding; ... }`)
        # is a separate node type from method_elem — an embedded interface
        # contributes only its methods to the composed interface, no method
        # name of its own, so it's recorded as a base class instead.
        for elem in find_nodes_by_type(type_node, "type_elem"):
            base_classes.append(extract_node_text(elem, src))
    else:
        # Struct embedding (`type Derived struct { Base; *Ptr; ... }`) is
        # this grammar's equivalent of inheritance/composition — an embedded
        # field_declaration has no "name" field (only "type"), unlike a
        # regular named field, which is how it's distinguished here.
        for field in find_nodes_by_type(type_node, "field_declaration"):
            if field.child_by_field_name("name") is not None:
                continue  # regular named field, not embedded
            type_field = field.child_by_field_name("type")
            if type_field is None:
                continue
            base_classes.append(extract_node_text(type_field, src))

    return ClassEntity(
        name=name,
        body_text=body_text,
        relative_file_path=relative_path,
        language="go",
        start_line=start_line,
        end_line=end_line,
        base_classes=base_classes,
        methods=methods,
        docstring=_extract_go_doc_comment(node, src),
        is_interface=is_interface,
    )


def _parse_go_imports(
    node: Node, src: bytes, relative_path: str
) -> list[ImportEntity]:
    """
    Parse a Go import_declaration into one ImportEntity per import_spec.

    Handles both a single import (`import "fmt"`, import_spec directly under
    import_declaration) and a grouped block (`import ( "fmt"; r "net/http" )`,
    multiple import_spec children nested inside an import_spec_list) — using
    find_nodes_by_type rather than direct children covers both shapes without
    needing to detect which one applies.
    """
    imports: list[ImportEntity] = []
    for spec in find_nodes_by_type(node, "import_spec"):
        imp = _parse_go_import_spec(spec, src, relative_path)
        if imp:
            imports.append(imp)
    return imports


def _parse_go_import_spec(
    node: Node, src: bytes, relative_path: str
) -> Optional[ImportEntity]:
    """
    Parse a single Go import_spec.

    Handles:
        "fmt"                 → module_path="fmt", no alias
        r "net/http"          → module_path="net/http", alias="r"
        _ "embed"             → module_path="embed", blank (side-effect) import
    """
    path_node = node.child_by_field_name("path")
    if not path_node:
        return None

    raw = extract_node_text(node, src)
    start_line, _ = get_node_line_range(node)

    # interpreted_string_literal's text includes the surrounding quotes.
    module_path = extract_node_text(path_node, src).strip('"')
    if not module_path:
        return None

    name_node = node.child_by_field_name("name")
    alias = extract_node_text(name_node, src) if name_node else None
    if alias == "_":
        alias = None  # blank import — no bound name, just runs init()

    imported_names = [module_path.split("/")[-1]]

    return ImportEntity(
        raw_statement=raw,
        module_path=module_path,
        relative_file_path=relative_path,
        language="go",
        start_line=start_line,
        imported_names=imported_names,
        is_relative=module_path.startswith("."),
        alias=alias,
    )


# ---------------------------------------------------------------------------
# Python-specific AST helpers
# ---------------------------------------------------------------------------

def _extract_python_module_docstring(root: Node, src: bytes) -> Optional[str]:
    """
    Extract the module-level docstring from a Python file's root node.

    WHY:
        Module docstrings describe the purpose of an entire file. Including
        them in the file-level metadata improves retrieval for queries like
        "find files related to authentication" where the module docstring
        (not any individual function) contains the best semantic signal.

    Args:
        root: Root node of the Python AST (the "module" node).
        src:  Raw source bytes.

    Returns:
        Module docstring text, or None if no docstring is present.
    """
    for child in root.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type == "string":
                    raw = extract_node_text(sub, src)
                    return raw.strip("'\"").strip('"""').strip("'''").strip()
    return None


def _extract_python_docstring_from_body(body_node: Node, src: bytes) -> Optional[str]:
    """
    Extract a docstring from a Python function or class body node.

    In Python, the docstring is the first statement in the body if that
    statement is an expression containing a string literal.

    Args:
        body_node: The "block" child node of a function or class definition.
        src:       Raw source bytes.

    Returns:
        Docstring text (stripped of quotes and surrounding whitespace), or None.
    """
    for child in body_node.children:
        if child.type == "expression_statement":
            for sub in child.children:
                if sub.type == "string":
                    raw = extract_node_text(sub, src)
                    # Strip triple and single/double quotes.
                    cleaned = raw.strip()
                    for q in ('"""', "'''", '"', "'"):
                        if cleaned.startswith(q) and cleaned.endswith(q):
                            cleaned = cleaned[len(q):-len(q)]
                            break
                    return cleaned.strip()
        # Docstring must be the first meaningful statement — stop after first non-trivial node.
        if child.type not in ("comment", "\n", "pass_statement"):
            break
    return None


def _parse_python_parameters(params_node: Node, src: bytes) -> list[str]:
    """
    Extract parameter names from a Python function's parameters node.

    Handles: positional, *args, **kwargs, keyword-only, and type-annotated params.
    Returns only the parameter names (not annotations or defaults) to keep
    the parameter list clean for metadata storage.

    Args:
        params_node: The "parameters" child of a function_definition.
        src:         Raw source bytes.

    Returns:
        List of parameter name strings, excluding 'self' and 'cls'.
    """
    params: list[str] = []
    SKIP = {"self", "cls", ",", "(", ")", "*", "**"}

    for child in params_node.children:
        if child.type in ("identifier",):
            name = extract_node_text(child, src)
            if name not in SKIP:
                params.append(name)
        elif child.type in ("typed_parameter", "typed_default_parameter",
                             "default_parameter"):
            name_child = child.child_by_field_name("name")
            if name_child:
                name = extract_node_text(name_child, src)
                if name not in SKIP:
                    params.append(name)
        elif child.type in ("list_splat_pattern", "dictionary_splat_pattern"):
            for sub in child.children:
                if sub.type == "identifier":
                    name = extract_node_text(sub, src)
                    if name not in SKIP:
                        params.append(f"*{name}" if child.type == "list_splat_pattern" else f"**{name}")

    return params


def _get_decorated_inner(decorated_node: Node) -> Optional[Node]:
    """
    Return the inner function or class from a decorated_definition node.

    In Python's tree-sitter grammar, a decorated function looks like:
        decorated_definition
            decorator
            decorator
            function_definition  ← this is what we want

    Args:
        decorated_node: A "decorated_definition" node.

    Returns:
        The inner "function_definition" or "class_definition" child, or None.
    """
    for child in decorated_node.children:
        if child.type in ("function_definition", "class_definition"):
            return child
    return None


def _extract_python_decorators(decorated_node: Node, src: bytes) -> list[str]:
    """
    Extract decorator names from a Python decorated_definition node.

    WHY:
        Decorators fundamentally change how a function behaves:
            @property         → getter method
            @staticmethod     → no instance binding
            @classmethod      → class-level method
            @pytest.mark.skip → test is skipped
            @app.route(...)   → Flask HTTP endpoint

        Storing decorator names in FunctionEntity metadata enables queries
        like "show me all Flask endpoints" or "which functions are properties?"

    Args:
        decorated_node: A "decorated_definition" node.
        src:            Raw source bytes.

    Returns:
        List of decorator name strings (e.g. ["property", "staticmethod"]).
    """
    decorators: list[str] = []
    for child in decorated_node.children:
        if child.type == "decorator":
            # Decorator text includes "@"; strip it for clean names.
            dec_text = extract_node_text(child, src).lstrip("@").strip()
            decorators.append(dec_text)
    return decorators