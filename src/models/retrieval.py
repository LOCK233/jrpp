from models.mol.utils_rails import get_top_k_module


class Retrieval:
    def __init__(self, config, model):
        self.config = config
        self.top_k_method = config["mol"]["top_k_method"]
        self.model = model

    def get_topk_related_items(self, item_embeddings, item_ids):
        return get_top_k_module(
            top_k_method=self.top_k_method,
            model=self.model,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
