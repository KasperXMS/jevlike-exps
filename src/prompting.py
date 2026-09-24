from .schema import MultipleChoiceSample


def option_label(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("Option index must be between 0 and 25")
    return chr(ord("A") + index)


def build_prompt(sample: MultipleChoiceSample) -> str:
    sections: list[str] = []
    if sample.context.strip():
        sections.append(f"Context:\n{sample.context.strip()}")
    sections.append(f"Question:\n{sample.question.strip()}")
    options = "\n".join(
        f"{option_label(index)}. {text.strip()}"
        for index, text in enumerate(sample.options)
    )
    sections.append(f"Options:\n{options}")
    sections.append("Answer:")
    return "\n\n".join(sections)

