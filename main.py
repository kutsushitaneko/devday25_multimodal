import gradio as gr
from app.config import Config
from app.embedding_service import EmbeddingService
from app.database_service import DatabaseService  
from app.search_service import SearchService
from app.ui.components import UIComponents
from app.ui.events import UIEvents
from app.search_query_generator import SearchQueryGenerator

def main():
    # 設定と基本サービスを初期化
    config = Config()
    db_connection = config.get_db_connection()
    cohere_client = config.get_cohere_client()
    search_query_generator = SearchQueryGenerator()
    
    # 各サービスを初期化
    embedding_service = EmbeddingService(cohere_client)
    database_service = DatabaseService(db_connection)
    search_service = SearchService(embedding_service, database_service, search_query_generator)
    
    # UIコンポーネントとイベントを初期化
    ui_components = UIComponents()
    ui_events = UIEvents(search_service)
    
    # Gradioインターフェースの作成
    with gr.Blocks(title="マルチモーダル画像検索") as demo:
        gr.Markdown("# マルチモーダル画像検索")
        gr.Markdown("画像を自然言語やアップロードした画像で検索できます。例: 「富士山と寺院」、「縞模様の猫」、「三匹の白い子猫」、「ホグワーツ魔法学校」、「上海のビル」、「2312.10997」、「search_queries_only」など")
        
        state = gr.State(None)
        
        # 検索セクションのUIコンポーネントを作成
        search_target, search_method, query_input, uploaded_image, search_button, clear_button, show_all_button, query_examples = ui_components.create_search_section()
        
        # 検索結果セクションのUIコンポーネントを作成
        vector_gallery, keyword_gallery = ui_components.create_results_section()
        
        # ページングセクションのUIコンポーネントを作成
        pagination_row, prev_button, page_info, next_button = ui_components.create_pagination_section()
        
        # 画像詳細セクションのUIコンポーネントを作成
        filename_text, similarity_text, caption_text, score_label = ui_components.create_detail_section()
        
        # クエリ詳細セクションのUIコンポーネントを作成
        executed_query_text, execute_query_button, executed_sql_text = ui_components.create_query_detail_section()
        
        # 高度な設定セクションのUIコンポーネントを作成
        vector_threshold, keyword_threshold, top_k_slider = ui_components.create_advanced_settings_section()
        
        # 各種イベントを登録
        ui_events.register_search_target_events(
            search_target, search_method, query_input, uploaded_image, query_examples, executed_sql_text
        )
        
        ui_events.register_search_method_events(
            search_method, query_input, uploaded_image, score_label, 
            executed_query_text, execute_query_button, search_target, query_examples
        )
        
        ui_events.register_search_button_events(
            search_button, query_input, uploaded_image, search_target, 
            search_method, top_k_slider, vector_threshold, keyword_threshold, 
            vector_gallery, keyword_gallery, filename_text, similarity_text, 
            caption_text, state, executed_query_text, executed_sql_text, execute_query_button, pagination_row
        )
        
        ui_events.register_execute_query_button_events(
            execute_query_button, executed_query_text, top_k_slider, keyword_threshold,
            vector_gallery, filename_text, similarity_text, caption_text, 
            state, executed_query_text, executed_sql_text, pagination_row
        )
        
        ui_events.register_clear_button_events(
            clear_button, query_input, uploaded_image, vector_gallery, 
            keyword_gallery, filename_text, similarity_text, caption_text, 
            state, executed_query_text, executed_sql_text, pagination_row
        )
        
        ui_events.register_show_all_button_events(
            show_all_button, top_k_slider, vector_gallery, 
            keyword_gallery, filename_text, similarity_text, caption_text, 
            state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button
        )
        
        ui_events.register_pagination_events(
            prev_button, next_button, top_k_slider, vector_gallery, page_info, state, keyword_gallery, prev_button, next_button
        )
        
        ui_events.register_gallery_selection_events(
            vector_gallery, keyword_gallery, state, 
            filename_text, similarity_text, caption_text
        )
        
        # アプリケーションの初期読み込み時のイベントを登録
        demo.load(
            fn=lambda: None,
            outputs=None
        ).then(
            fn=ui_events.show_all_images,
            inputs=[top_k_slider],
            outputs=[vector_gallery, keyword_gallery, filename_text, similarity_text, 
                     caption_text, state, executed_query_text, executed_sql_text, pagination_row, page_info, prev_button, next_button]
        )
    
    # アプリケーションの起動
    demo.launch(share=False, inbrowser=True)

if __name__ == "__main__":
    main() 