import re


def extract_sections(text):
    """Parse raw SAT text into structured sections"""
    sections = {
        'passage': '',
        'question': '',
        'choices': [],
        'answer_letter': ''
    }

    answer_part = text.split('Answer:')[-1].strip()
    sections['answer_letter'] = answer_part[0] if answer_part else ''

    content = text.split(
        'SAT READING COMPREHENSION TEST')[-1].split('Answer:')[0]
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]

    passage_lines = []
    for line in blocks:
        if line.startswith('Question'):
            break
        passage_lines.append(line)
    sections['passage'] = '\n'.join(passage_lines).strip()

    for block in blocks:
        if block.startswith('Question'):
            lines = block.split('\n')
            question_lines = []
            choice_lines = []

            for line in lines[1:]:
                if re.match(r'^[A-D]\)', line.strip()):
                    choice_lines.append(line.strip())
                else:
                    question_lines.append(line.strip())

            sections['question'] = ' '.join(question_lines).strip()
            sections['choices'] = choice_lines

    return sections
