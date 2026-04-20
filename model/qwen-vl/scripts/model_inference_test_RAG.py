import os
import logging
from typing import Optional, Dict, List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import *

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ===================== 核心配置 =====================
CHANNEL_MAP = {
    "综合诊疗决策通道": "comprehensive_treatment",
    "影像特征与解读通道": "imaging_features",
    "生化指标与临床阈值通道": "biochemistry_thresholds",
    "量表评分与活动度分级通道": "assessment_scales",
    "临床症状与鉴别通道": "clinical_symptoms"
}

ANATOMY_MAP = {
    "骶髂关节": "SACROILIAC_JOINT",
    "脊柱": "SPINE",
    "髋关节": "HIP_JOINT",
    "外周关节": "PERIPHERAL_JOINT",
    "足跟": "HEEL",
    "胸骨": "STERNUM",
    "骨盆": "PELVIS",
    "肋骨": "RIB",
    "肩关节": "SHOULDER_JOINT",
    "肘关节": "ELBOW_JOINT",
    "腕关节": "WRIST_JOINT",
    "膝关节": "KNEE_JOINT",
    "踝关节": "ANKLE_JOINT"
}

CHANNEL_TOP_K = {
    "综合诊疗决策通道": 5,
    "影像特征与解读通道": 3,
    "生化指标与临床阈值通道": 2,
    "量表评分与活动度分级通道": 2,
    "临床症状与鉴别通道": 2
}

SCORE_THRESHOLD = 0.5
MAX_CONTEXT_LENGTH = 8000  # 全文较长，适当扩容
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# ===================== 五通道RAG核心类 =====================
class ASChannelRAG:
    def __init__(self):
        self.embeddings = self._init_embedding()
        self.channel_dbs = self._load_all_channels()

    def _init_embedding(self) -> HuggingFaceEmbeddings:
        logger.info("加载文本向量Embedding模型...")
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': DEVICE}
            )
        except Exception as e:
            logger.error(f"❌ Embedding模型加载失败！错误: {str(e)}")
            raise RuntimeError("Embedding模型初始化失败，系统终止运行") from e

    def _load_all_channels(self) -> Dict[str, Chroma]:
        dbs = {}
        logger.info("开始加载五通道独立向量库...")
        for ch_name, en_name in CHANNEL_MAP.items():
            db_path = os.path.join(CHROMA_DB_PATH, en_name)
            if os.path.exists(db_path):
                try:
                    dbs[ch_name] = Chroma(
                        collection_name=en_name,
                        persist_directory=db_path,
                        embedding_function=self.embeddings
                    )
                    logger.info(f"✅ 通道加载成功：{ch_name}")
                except Exception as e:
                    logger.error(f"❌ 通道加载失败：{ch_name}，错误：{str(e)}")
            else:
                logger.warning(f"⚠️ 通道路径不存在：{db_path}")
        return dbs

    def _get_db(self, channel: str) -> Optional[Chroma]:
        if channel in self.channel_dbs:
            return self.channel_dbs[channel]
        logger.error(f"通道 '{channel}' 未加载！")
        return None

    def _standard_anatomy(self, anatomy: str) -> str:
        if not anatomy or anatomy == "UNKNOWN":
            return "UNKNOWN"
        return ANATOMY_MAP.get(anatomy, anatomy.strip())

    def _retrieve_with_fallback(self, channel: str, query: str, anatomy: str = "UNKNOWN") -> List[Dict]:
        db = self._get_db(channel)
        if not db:
            return []
        
        k = CHANNEL_TOP_K[channel]
        all_results = []

        # 仅影像通道多查询检索
        if channel == "影像特征与解读通道" and anatomy != "UNKNOWN":
            queries = [query, f"{anatomy} {query}"]
        else:
            queries = [query]

        for q in queries:
            try:
                if channel == "影像特征与解读通道" and anatomy != "UNKNOWN":
                    res_with_score = db.similarity_search_with_score(query=q, k=k, filter={"anatomy": anatomy})
                else:
                    res_with_score = db.similarity_search_with_score(query=q, k=k)

                # 🔥 核心修复：清理source_block中的换行符，去掉多余回车，保留完整内容
                filtered = [
                    {
                        # 删掉冗余重复的【通道名】前缀
                        "content": doc.page_content.replace(f"【{channel}】 ", ""),
                        # 清理换行/回车，保证来源显示在同一行
                        "source": doc.metadata.get("source_block", "AS诊疗知识库").replace("\n", "").replace("\r", "").strip(),
                        "page": ""
                    }
                    for doc, score in res_with_score if score >= SCORE_THRESHOLD
                ]
                all_results.extend(filtered)

                # 影像空结果回退
                if channel == "影像特征与解读通道" and not filtered and anatomy != "UNKNOWN":
                    logger.warning(f"影像通道{anatomy}无结果，执行通用检索")
                    fallback_res = db.similarity_search_with_score(query=q, k=k)
                    fallback_filtered = [
                        {
                            "content": doc.page_content.replace(f"【{channel}】 ", ""),
                            "source": doc.metadata.get("source_block", "AS诊疗知识库").replace("\n", "").replace("\r", "").strip(),
                            "page": ""
                        }
                        for doc, score in fallback_res if score >= SCORE_THRESHOLD
                    ]
                    all_results.extend(fallback_filtered)
            except Exception as e:
                logger.error(f"检索异常：{str(e)}")

        # 顺序去重，保留排名
        unique_results = []
        seen_content = set()
        for item in all_results:
            if item["content"] not in seen_content:
                seen_content.add(item["content"])
                unique_results.append(item)

        return unique_results

    def retrieve(
        self,
        query: str,
        imaging: Optional[str] = None,
        biochemistry: Optional[str] = None,
        scale: Optional[str] = None,
        symptom: Optional[str] = None,
        anatomy: Optional[str] = "UNKNOWN"
    ) -> Dict[str, List[Dict]]:
        results = {}
        anatomy_std = self._standard_anatomy(anatomy)

        if imaging:
            results["影像特征与解读通道"] = self._retrieve_with_fallback("影像特征与解读通道", imaging, anatomy_std)
        if biochemistry:
            results["生化指标与临床阈值通道"] = self._retrieve_with_fallback("生化指标与临床阈值通道", biochemistry)
        if scale:
            results["量表评分与活动度分级通道"] = self._retrieve_with_fallback("量表评分与活动度分级通道", scale)
        if symptom:
            results["临床症状与鉴别通道"] = self._retrieve_with_fallback("临床症状与鉴别通道", symptom)
        
        results["综合诊疗决策通道"] = self._retrieve_with_fallback("综合诊疗决策通道", query)

        logger.info(f"检索完成，有效通道：{list(results.keys())}")
        return results

    def format_context(self, retrieve_results: Dict[str, List[Dict]]) -> str:
        context_parts = []
        sort_order = [
            "综合诊疗决策通道",
            "影像特征与解读通道",
            "生化指标与临床阈值通道",
            "量表评分与活动度分级通道",
            "临床症状与鉴别通道"
        ]

        for channel in sort_order:
            items = retrieve_results.get(channel, [])
            if not items:
                continue

            context_parts.append(f"\n===== {channel} =====")
            for idx, item in enumerate(items, 1):
                # ✅ 唯一修改：来源纯后置，不破坏语义，完整保留所有内容
                context_parts.append(f"{idx}. {item['content']} | [来源: {item['source']}]")

        # 上下文长度限制
        final_context = "\n".join(context_parts).strip()
        if len(final_context) > MAX_CONTEXT_LENGTH:
            final_context = final_context[:MAX_CONTEXT_LENGTH] + "...(已截断超长上下文)"
            logger.warning("上下文超长，已自动截断")

        return final_context if final_context else "未检索到相关强直性脊柱炎诊疗知识"

# ===================== 测试入口 =====================
if __name__ == "__main__":
    logger.info("启动强直性脊柱炎五通道医疗RAG系统...")
    rag = ASChannelRAG()

    TEST_QUERY = "强直性脊柱炎诊断与治疗方案"
    TEST_IMAGING = "骶髂关节骨髓水肿、骨侵蚀"
    TEST_BIOCHEMISTRY = "CRP升高，血沉增快"
    TEST_SCALE = "BASDAI评分5分"
    TEST_SYMPTOM = "晨僵大于30分钟，下腰背部疼痛"
    TEST_ANATOMY = "骶髂关节"

    rag_results = rag.retrieve(
        query=TEST_QUERY, imaging=TEST_IMAGING, biochemistry=TEST_BIOCHEMISTRY,
        scale=TEST_SCALE, symptom=TEST_SYMPTOM, anatomy=TEST_ANATOMY
    )
    final_context = rag.format_context(rag_results)
    
    print("\n" + "="*80)
    print("📋 专业医疗RAG参考上下文（无换行·完整溯源）")
    print("="*80)
    print(final_context)