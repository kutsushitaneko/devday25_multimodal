import oci
import cohere
import oracledb
from PIL import Image
from io import BytesIO
import base64
import os
import array
import time
import glob
import sys
from dotenv import load_dotenv, find_dotenv

def image_to_base64_data_url(image_data):
    """画像データをBase64エンコードしてData URLに変換"""
    img = Image.open(BytesIO(image_data))
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{img_base64}"
    return data_url

def get_image_caption(generative_ai_inference_client, image_data):
    """画像データからキャプションを生成する関数"""
    #PROMPT = "画像に描かれているものを詳しく説明してください。"
    PROMPT = "画像からすべてのテキストを抽出して、画像に描かれているものを教えてください。固有の名称を教えてください。画像を説明してください。"
    content1 = oci.generative_ai_inference.models.TextContent()
    content1.text = PROMPT
    content2 = oci.generative_ai_inference.models.ImageContent()
    image_url = oci.generative_ai_inference.models.ImageUrl()
    image_url.url = image_to_base64_data_url(image_data)
    content2.image_url = image_url
    message = oci.generative_ai_inference.models.UserMessage()
    message.content = [content1,content2]

    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = [message]
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.num_generations = 1
    chat_request.max_tokens = 600
    chat_request.is_stream = False
    chat_request.temperature = 0.70
    chat_request.top_p = 0.7
    chat_request.top_k = -1
    chat_request.frequency_penalty = 0.5
    chat_request.presence_penalty = 0.5

    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=MLLM_MODEL_ID)
    chat_detail.compartment_id = COMPARTMENT_ID
    chat_detail.chat_request = chat_request

    chat_response = generative_ai_inference_client.chat(chat_detail)

    # Print result
    print("************************** Chat Result *******************************")
    print(vars(chat_response))
    print("************************** Generated Caption**************************")
    
    # 正しいレスポンス構造からテキストを取得
    if hasattr(chat_response, 'data') and hasattr(chat_response.data, 'chat_response'):
        if hasattr(chat_response.data.chat_response, 'choices') and len(chat_response.data.chat_response.choices) > 0:
            choice = chat_response.data.chat_response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                for content in choice.message.content:
                    if hasattr(content, 'text'):
                        print(content.text)
                        # VARCHAR2(4000)の制限を考慮して、キャプションを4000文字以内に切り詰める
                        return content.text[:4000] if len(content.text) > 4000 else content.text
    
    # 上記の方法でテキストを取得できない場合は、代替方法を試す
    print("標準的な方法でテキストを取得できませんでした。代替方法を試みます。")
    try:
        # レスポンスの構造を確認
        response_data = chat_response.data
        if hasattr(response_data, 'chat_response'):
            chat_response_data = response_data.chat_response
            if hasattr(chat_response_data, 'choices') and len(chat_response_data.choices) > 0:
                first_choice = chat_response_data.choices[0]
                if hasattr(first_choice, 'message'):
                    message = first_choice.message
                    if hasattr(message, 'content') and len(message.content) > 0:
                        for content_item in message.content:
                            if hasattr(content_item, 'text'):
                                # VARCHAR2(4000)の制限を考慮して、キャプションを4000文字以内に切り詰める
                                return content_item.text[:4000] if len(content_item.text) > 4000 else content_item.text
    except Exception as e:
        print(f"代替方法でもエラーが発生しました: {str(e)}")
    
    # すべての方法が失敗した場合
    return "画像の説明を生成できませんでした。"

def get_image_embedding(co, image_data):
    """画像データからCohere Embed 3を使用して埋め込みベクトルを生成"""
    # 画像データをBase64エンコードしてData URLに変換
    data_url = image_to_base64_data_url(image_data)
    
    # Cohere APIを使用して画像の埋め込みベクトルを取得
    response = co.embed(
        images=[data_url],
        model="embed-v4.0",
        input_type="image",
        embedding_types=["float"],
    )
    
    return response.embeddings.float[0]

def get_text_embedding(co, text):
    """テキストからCohere Embed 3を使用して埋め込みベクトルを生成"""
    response = co.embed(
        texts=[text],
        model="embed-v4.0",
        input_type="search_document"
    )
    
    return response.embeddings[0]

def is_image_registered(db_connection, file_name):
    """画像が既にデータベースに登録されているかチェック"""
    cursor = db_connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM IMAGES WHERE file_name = :1", [file_name])
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        cursor.close()

def insert_image_to_db(generative_ai_inference_client, cohere_client, db_connection, image_data, file_name):
    """画像とその説明文をOracle Databaseに挿入"""
    caption = get_image_caption(generative_ai_inference_client, image_data)
    # 画像とテキストの埋め込みベクトルを取得
    image_embedding = array.array('f', get_image_embedding(cohere_client, image_data))
    caption_embedding = array.array('f', get_text_embedding(cohere_client, caption))
    
    cursor = db_connection.cursor()
    try:
        # 画像データを挿入
        cursor.execute("""
            INSERT INTO IMAGES (file_name, caption, caption_embedding, image_data, image_embedding)
            VALUES (:1, :2, :3, :4, :5)
        """, (
            file_name,
            caption,
            caption_embedding,
            image_data,
            image_embedding
        ))
        
        db_connection.commit()
        print(f"画像 '{file_name}' が正常に挿入されました。")
        return True
    except Exception as e:
        print(f"画像 '{file_name}' の挿入中にエラーが発生しました: {str(e)}")
        raise
    finally:
        cursor.close()

if __name__ == "__main__":
    # 処理開始時間を記録
    start_time = time.time()
    
    # 環境変数を読み込む
    load_dotenv(find_dotenv())
    
    # 必要な環境変数のリスト
    required_env_vars = [
        "COHERE_API_KEY",
        "TNS_ADMIN",
        "DB_USER",
        "DB_PASSWORD",
        "DB_DSN",
        "OCI_CONFIG_PROFILE",
        "OCI_REGION",
        "OCI_COMPARTMENT_ID",
        "OCI_GENAI_MLLM_MODEL_ID"
    ]
    
    # 環境変数の存在確認
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("エラー: 以下の環境変数が設定されていません:")
        for var in missing_vars:
            print(f"  - {var}")
        exit(1)
    
    # 環境変数から設定を読み込む
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    USERNAME = os.getenv("DB_USER")
    PASSWORD = os.getenv("DB_PASSWORD")
    DSN = os.getenv("DB_DSN")
    
    # 画像ディレクトリのパス
    IMAGE_DIR = "images"
    
    # 統計情報の初期化
    total_images = 0
    already_registered = 0
    newly_registered = 0
    failed_registrations = 0

    # OCI Config の設定
    CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE")
    config = oci.config.from_file(file_location='~/.oci/config', profile_name=CONFIG_PROFILE)
    config["region"] = os.getenv("OCI_REGION")

    COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID") 
    MLLM_MODEL_ID = os.getenv("OCI_GENAI_MLLM_MODEL_ID")
    
    try:
        # OCI GenAIクライアントを初期化
        generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240)) 

        # Cohereクライアントを初期化
        cohere_client = cohere.Client(api_key=COHERE_API_KEY)

        # Oracle接続を確立
        db_connection = oracledb.connect(user=USERNAME, password=PASSWORD, dsn=DSN)
        print("データベース接続成功!")
        
        # images ディレクトリ内のすべての画像ファイルを取得
        image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        total_images = len(image_files)
        
        print(f"処理対象の画像ファイル数: {total_images}")
        
        # 各画像ファイルを処理
        for image_path in image_files:
            file_name = os.path.basename(image_path)
            
            # 既に登録されているかチェック
            if is_image_registered(db_connection, file_name):
                print(f"画像 '{file_name}' は既に登録されています。スキップします。")
                already_registered += 1
                continue
            
            print(f"画像 '{file_name}' を処理中...")
            
            try:
                # 画像データを読み込む
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                # 画像データを挿入
                if insert_image_to_db(generative_ai_inference_client, cohere_client, db_connection, image_data, file_name):
                    newly_registered += 1
                else:
                    failed_registrations += 1
                    
            except Exception as e:
                print(f"画像 '{file_name}' の処理中にエラーが発生しました: {str(e)}")
                raise
        
        # 処理時間を計算
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 統計情報を表示
        print("\n===== 処理結果サマリー =====")
        print(f"ディレクトリ内の画像ファイル総数: {total_images}")
        print(f"既に登録済みの画像数: {already_registered}")
        print(f"今回新規登録した画像数: {newly_registered}")
        print(f"登録に失敗した画像数: {failed_registrations}")
        print(f"処理時間: {processing_time:.2f} 秒")
        
    except Exception as e:
        print("エラーが発生しました！")
        print(f"エラーの種類: {type(e).__name__}")
        print(f"エラーの内容: {str(e)}")
        print("エラーが発生したため処理を中止します。")
        sys.exit(1)
    
    finally:
        # DB接続を閉じる
        if 'db_connection' in locals():
            db_connection.close()
