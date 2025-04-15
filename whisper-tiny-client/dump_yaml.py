import yaml

chat_messages = [
    {
        "role": "system",
        "content": "你是一个名为讲住英的虚拟人，由杭州电子科技大学开发和微调的人工智能助手。",
    },
    {
        "role": "user",
        "content": "hello."
    }
]

# with open('chat_messages.yaml', 'w', encoding="utf-8") as file:
#     yaml.dump(chat_messages, file, allow_unicode=True)
import os
with open('chat_messages.yml', 'r', encoding="utf-8") as stream:
    try:
        value = yaml.safe_load(stream)
        pass
    except yaml.YAMLError as ex:
        print(ex)