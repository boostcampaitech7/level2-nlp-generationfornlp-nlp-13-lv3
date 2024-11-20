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
