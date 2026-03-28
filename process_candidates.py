import json
import time
# Импортируем наш класс из папки include
from include.main import CandidateAnalyzer

def process_candidates():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   inVision U — Smart Candidate Analyzer             ║")
    print("║   Decentrathon 5.0 | AI inDrive                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    try:
        analyzer = CandidateAnalyzer()
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        return

    # Читаем моковые данные
    try:
        with open('mock_data.json', 'r', encoding='utf-8') as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print("❌ Файл mock_data.json не найден в корневой папке!")
        return

    final_results = []
    print(f"- Начинаем обработку {len(candidates)} эссе...\n")

    for person in candidates:
        candidate_id = person.get('id', 'Неизвестно')
        print(f"🔄 Анализ кандидата ID: {candidate_id} ...")
        
        essay_text = person.get('essay', '')
        if not essay_text:
            print("  ⚠️ Пустое эссе, пропускаем.")
            continue

        # Вся магия происходит здесь
        analysis = analyzer.analyze_essay(essay_text)
        
        report = {
            "candidate_id": candidate_id,
            "analysis": analysis
        }
        final_results.append(report)
        
        # Небольшая пауза, чтобы API не ругался на частые запросы (хоть Groq и быстрый)
        time.sleep(1)

    # Сохраняем результаты
    with open('final_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print("\n✅ Анализ завершен! Результаты сохранены в файл final_results.json")

if __name__ == "__main__":
    process_candidates()