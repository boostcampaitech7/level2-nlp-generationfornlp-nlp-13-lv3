def generate_chat_template(messages):
    """
    Generate a chat template string from message data.
    """
    template = []
    for message in messages:
        if message["role"] == "system":
            template.append(f"<system>{message['content']}</system>")
        elif message["role"] == "user":
            template.append(f"<start_of_turn>user\n{message['content']}<end_of_turn>\n<start_of_turn>model\n")
        elif message["role"] == "assistant":
            template.append(f"{message['content']}<end_of_turn>\n")
    return "".join(template)


def get_chat_template():

    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% endif %}"
        "{% if system_message is defined %}"
        "{{ system_message }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ content + '<end_of_turn>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    )

    return template
