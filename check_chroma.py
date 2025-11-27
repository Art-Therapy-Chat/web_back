"""
ChromaDB 상태 확인 스크립트
"""

import sys
sys.path.append('/web_back')

from dotenv import load_dotenv
load_dotenv()

from embeddings import vectorstore

print("=" * 80)
print("🔍 ChromaDB 상태 확인")
print("=" * 80)

try:
    # 전체 문서 개수 확인
    collection = vectorstore._collection
    count = collection.count()
    
    print(f"\n📊 총 문서 개수: {count}")
    
    if count == 0:
        print("\n⚠️  ChromaDB가 비어있습니다!")
        print("   data/ 폴더의 txt 파일들이 임베딩되지 않았을 수 있습니다.")
    else:
        print(f"\n✅ {count}개의 문서가 저장되어 있습니다.")
        
        # 샘플 문서 몇 개 가져오기
        print("\n📄 샘플 문서 (최대 3개):")
        print("-" * 80)
        
        results = collection.get(limit=3, include=['documents', 'metadatas'])
        
        for idx, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
            print(f"\n문서 {idx}:")
            print(f"  메타데이터: {meta}")
            print(f"  내용 (처음 200자): {doc[:200]}...")
            print("-" * 80)
        
        # 테스트 검색 - "나무" 키워드로
        print("\n🔍 테스트 검색 - '나무' 키워드:")
        print("-" * 80)
        
        test_results = vectorstore.similarity_search("나무가 크다", k=3)
        
        if test_results:
            print(f"\n✅ {len(test_results)}개의 관련 문서를 찾았습니다:")
            for idx, doc in enumerate(test_results, 1):
                print(f"\n결과 {idx}:")
                print(f"  메타데이터: {doc.metadata}")
                print(f"  내용 (처음 200자): {doc.page_content[:200]}...")
                print("-" * 80)
        else:
            print("\n⚠️  검색 결과가 없습니다.")
        
        # 테스트 검색 - "tree" 영어 키워드로
        print("\n🔍 테스트 검색 - 'tree' 영어 키워드:")
        print("-" * 80)
        
        test_results_en = vectorstore.similarity_search("the tree is big", k=3)
        
        if test_results_en:
            print(f"\n✅ {len(test_results_en)}개의 관련 문서를 찾았습니다:")
            for idx, doc in enumerate(test_results_en, 1):
                print(f"\n결과 {idx}:")
                print(f"  메타데이터: {doc.metadata}")
                print(f"  내용 (처음 200자): {doc.page_content[:200]}...")
                print("-" * 80)
        else:
            print("\n⚠️  검색 결과가 없습니다.")

except Exception as e:
    print(f"\n❌ 에러 발생: {str(e)}")
    import traceback
    print("\n스택 트레이스:")
    print(traceback.format_exc())

print("\n" + "=" * 80)
print("✅ 확인 완료")
print("=" * 80)
