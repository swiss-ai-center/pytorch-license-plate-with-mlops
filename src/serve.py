from mlem.api import serve

serve(
    model="models/model_stack",
    server="streamlit",
    request_serializer="torch_image",
    response_serializer="dict",
)
