"""검색 품질 평가 스크립트.

data/jobs.txt의 공고들로 FAISS 인덱스를 만들고,
data/eval_set.json의 질문-정답 쌍으로 검색 정확도(hit@k)를 측정한다.

사용법:
    python 09_eval.py            # hit@1, hit@3 측정
    python 09_eval.py --show     # 각 질문의 검색 결과까지 출력
"""

import argparse
import json
import re
import sys

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 윈도우 콘솔 한글 출력 대비
sys.stdout.reconfigure(encoding="utf-8")

JOBS_PATH = "data/jobs.txt"
EVAL_PATH = "data/eval_set.json"


def load_jobs():
    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    jobs = [j.strip() for j in content.split("---") if j.strip()]
    docs = [Document(page_content=j) for j in jobs]
    return docs


def job_id(doc):
    """공고 본문에서 [공고N] 태그를 꺼낸다."""
    m = re.search(r"\[(공고\d+)\]", doc.page_content)
    return m.group(1) if m else "(태그 없음)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="질문별 검색 결과 출력")
    parser.add_argument("--k", type=int, default=3, help="검색 상위 k (기본 3)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama 임베딩 모델 (기본 nomic-embed-text)")
    args = parser.parse_args()

    docs = load_jobs()
    print(f"공고 수: {len(docs)}건")

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)["eval"]
    print(f"평가 질문 수: {len(eval_set)}개\n")

    print(f"임베딩 모델: {args.embed_model}")
    embeddings = OllamaEmbeddings(model=args.embed_model)
    vectorstore = FAISS.from_documents(docs, embeddings)

    hit1 = 0
    hitk = 0
    for item in eval_set:
        q = item["question"]
        expected = set(item["expected"])
        results = vectorstore.similarity_search(q, k=args.k)
        got = [job_id(d) for d in results]

        top1_ok = got[0] in expected
        topk_ok = any(g in expected for g in got)
        hit1 += top1_ok
        hitk += topk_ok

        mark = "O" if topk_ok else "X"
        if args.show or not topk_ok:
            print(f"[{mark}] {q}")
            print(f"    기대: {sorted(expected)} / 검색: {got}")

    n = len(eval_set)
    print(f"\n=== 결과 ===")
    print(f"hit@1: {hit1}/{n} ({hit1/n:.0%})")
    print(f"hit@{args.k}: {hitk}/{n} ({hitk/n:.0%})")


if __name__ == "__main__":
    main()
