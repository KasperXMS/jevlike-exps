from .schema import MultipleChoiceSample


def option_label(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("Option index must be between 0 and 25")
    return chr(ord("A") + index)


def _question_sections(sample: MultipleChoiceSample) -> list[str]:
    sections: list[str] = []
    if sample.context.strip():
        sections.append(f"Context:\n{sample.context.strip()}")
    sections.append(f"Question:\n{sample.question.strip()}")
    options = "\n".join(
        f"{option_label(index)}. {text.strip()}"
        for index, text in enumerate(sample.options)
    )
    sections.append(f"Options:\n{options}")
    return sections


def build_prompt(sample: MultipleChoiceSample) -> str:
    sections = _question_sections(sample)
    sections.append("Answer:")
    return "\n\n".join(sections)


def build_prompt_parts(sample: MultipleChoiceSample) -> tuple[str, str]:
    """Split the standard prompt immediately before option A for KV branching."""
    leading_sections: list[str] = []
    if sample.context.strip():
        leading_sections.append(f"Context:\n{sample.context.strip()}")
    leading_sections.append(f"Question:\n{sample.question.strip()}")
    prefix = "\n\n".join(leading_sections) + "\n\nOptions:\n"
    options = "\n".join(
        f"{option_label(index)}. {text.strip()}"
        for index, text in enumerate(sample.options)
    )
    return prefix, options + "\n\nAnswer:"


def build_exact_text_prompt(sample: MultipleChoiceSample) -> str:
    sections = _question_sections(sample)
    sections.append("Answer with the exact text of the correct option:")
    return "\n\n".join(sections)
