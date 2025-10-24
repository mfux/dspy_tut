---
applyTo: "**/*.py"
---
# Python Code Style Guidelines

## Naming Conventions

- **Modules & packages**: `snake_case`
- **Classes**: `PascalCase`
- **Functions & variables**: `snake_case`
- **Constants** (module-level, immutable): `ALL_CAPS`
- **Private methods** prefixed with `_`

## Style Examples

Here are some examples of what I consider good code style.
Mind the style of naming variables, structuring the code into steps, and mindful use of abbreviations.

```python
def search(substring, words):
    result = []
    for word in words:
        if substring in word:
            result.append(word)
    return result
```

```python
def extract_property(prop):
    return [prop(word) for word in sent]
```

## Quotes

- Always use double quotes for strings (except when nested)

## Type Annotations
- Type Hints are encouraged
- Use typing module for complex types (List, Dict, Optional, Union, etc.)

## Docstrings

- **Style**: Sphinx/ReST (triple-double quotes).
- **Structure**:
  1. One-line summary.
  2. Blank line.
  3. Detailed description or numbered steps.
  4. Parameter/return/raises sections.
- **Example (class)**:
  ```python
  class ExaBGPProcessor:
      """
      Consume & parse an ExaBGP JSON stream from stdin.

      Processing steps:
      1. Convert raw JSON to dict.
      2. Derive message type/subtype.
      3. Validate required fields.
      4. Extract and enqueue config updates.

      :param term_on_shutdown: if True, send SIGINT on shutdown msg.
      :Keyword Arguments:
          * max_paths_per_next_hop* — max announced paths per next hop.
          * next_hop_filter* — lambda-string to filter next hops.
          ...
      """
      ...
  ```
- **Example (function)**:
  ```python
  def process_msg(raw_msg):
      """
      Parse, validate and apply a single ExaBGP message.

      :param raw_msg: JSON‐encoded message from ExaBGP.
      :return: reformatted, validated dict.
      :raises ValueError: if mandatory fields are missing or malformed.
      """
      ...
  ```
- For functions, use a single line for simple cases:
  ```python
  def add(a: int, b: int) -> int:
      """Return the sum of a and b."""
      return a + b
  ```

## Comment Style

- **When to comment**:
  - Non-obvious logic blocks
  - High-level steps in a function
  - Important design decisions
  - short comments that increase readability
- **How to comment**:
  - For High-level steps, use a brief summary.
    - Use full sentences.
    - Start with a capital letter; end with a period.
    - **Example**:
      ```python
      # Check for well-formed JSON by decoding.
      msg = p_helpers.parse_msg_json(raw_msg)

      # If this is a shutdown notification, send SIGINT.
      if msg_type == "notification" and msg_subtype == "shutdown":
          os.kill(os.getpid(), signal.SIGINT)
      ```
  - Short comments to increase readability or add context
    - Use inline comments sparingly.
    - Keep them concise and relevant.
    - All lower case, no punctuation, e.g. `# parse JSON`, `# check for shutdown`
    - **Example**:
      ```python
      # set up variables
      messages = list()
      message_obj = None
      line_idx = 0

      # iterate over the lines
      with open(file_path, "r", encoding="utf-8") as file:
          for line in file:
              # ignore heading line
              line_idx += 1
      ```
      ```python
      def process_event(event: dict):
          payload = event.get("payload", {})  # legacy events don’t include payload
      ```

## Tooling & Quality

- Use **pytest** as primary test runner
- Enforce style with `pre-commit run`

## Best Practices
- For complex list processing operations, prefer generator expressions to save memory
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- Use dataclasses for data containers.
- Implement special methods as needed (`__str__`, `__repr__`, etc.).
- Use properties instead of getter/setter methods when appropriate.

## Libraries
- Always use pathlib for file system paths.


## Design Principles
- **Keep It Simple (KISS)**  
  - Limit functions to a single, clear task (≈20–30 lines).  
  - Don’t introduce abstractions or patterns unless they solve a demonstrated need.  
  - Prefer clear, straightforward logic over “clever” one-liners.

- **Don’t Repeat Yourself (DRY)**  
  - Before copying code, search for existing helpers or utilities.  
  - Extract repeated logic into well-named functions or modules.  
  - Share common setup and teardown in fixtures or base classes.

- **Separation of Concerns**  
  - Group code by layer: e.g. data access, business logic, and presentation each live in their own modules or packages.  
  - Avoid “god modules”—if a file grows beyond ~200 lines or touches multiple concerns, split it.  


