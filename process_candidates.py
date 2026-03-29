import json
import time
import sys
from include.main import CandidateAnalyzer

def print_summary_table(results: list):
    """Выводит итоговую сводную таблицу всех кандидатов по завершении."""
    print(f"\n\n{'━' * 80}")
    print(f"  📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'━' * 80}")
    print(f"  {'ID':<12} | {'Lead':>6} | {'Motiv':>6} | {'Grow':>6} | {'AI Risk':>10}")
    print(f"  {'─' * 12}-+-{'─' * 6}-+-{'─' * 6}-+-{'─' * 6}-+-{'─' * 12}")

    scored = []
    for r in results:
        cid = r["candidate_id"]
        a = r["analysis"]
        s = a.get("scores", {})

        lead = s.get("leadership", 0)
        moti = s.get("motivation", 0)
        grow = s.get("growth_path", 0)
        ai_r = s.get("ai_risk", 0)
        ai_l = a.get("ai_risk_level", "Unknown")

        lead_str = f"{lead:.2f}" if isinstance(lead, (int, float)) else "N/A"
        moti_str = f"{moti:.2f}" if isinstance(moti, (int, float)) else "N/A"
        grow_str = f"{grow:.2f}" if isinstance(grow, (int, float)) else "N/A"
        
        emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(ai_l, "⚪")
        ai_r_str = f"{emoji} {ai_l}"

        print(f"  {cid:<12} | {lead_str:>6} | {moti_str:>6} | {grow_str:>6} | {ai_r_str:>10}")

        # Считаем композитный балл (рейтинг)
        if isinstance(lead, (int, float)) and isinstance(ai_r, (int, float)):
            composite = (lead * 0.4 + moti * 0.3 + grow * 0.3) * (1 - ai_r)
            scored.append((cid, composite))

    print(f"{'━' * 80}")

    if scored:
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  🏆 РЕКОМЕНДОВАННЫЙ SHORTLIST:")
        for i, (cid, score) in enumerate(scored[:3], 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f" {i}.")
            print(f"     {medal} {cid} (Score: {score:.2f})")
    print()


def process_candidates():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   inVision U — Smart Candidate Analyzer              ║")
    print("║   Decentrathon 5.0 | AI inDrive                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    print("🚀 Инициализация анализатора (загрузка NER-моделей)...")
    try:
        analyzer = CandidateAnalyzer()
    except ValueError as e:
        print(f"❌ Ошибка инициализации: {e}")
        return
    except Exception as e:
        print(f"❌ Системная ошибка: {e}")
        return

    # Читаем моковые данные
    try:
        with open('mock_data.json', 'r', encoding='utf-8') as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print("❌ Ошибка: Файл mock_data.json не найден в корневой папке!")
        return
    except json.JSONDecodeError:
        print("❌ Ошибка: mock_data.json содержит невалидный JSON!")
        return

    total_candidates = len(candidates)
    if total_candidates == 0:
        print("⚠️ Список кандидатов пуст.")
        return

    final_results = []
    print(f"\n📊 Начинаем обработку {total_candidates} эссе через Groq API...\n")

    start_time = time.time()

    for i, person in enumerate(candidates, 1):
        candidate_id = person.get('id', f'UNKNOWN_{i}')
        essay_text = person.get('essay', '')
        
        # Красивый вывод статуса обработки в одну строку
        sys.stdout.write(f"🔄 [{i}/{total_candidates}] Анализ {candidate_id}... ")
        sys.stdout.flush()
        
        if not essay_text.strip():
            print(f"⚠️ Пропущен (пустое эссе)")
            continue

        try:
            # Запуск анализа
            analysis = analyzer.analyze_essay(essay_text)
            
            # Проверяем на внутренние ошибки от LLM (наша заглушка возвращает ключ 'error')
            if "error" in analysis:
                print(f"❌ Ошибка LLM: {analysis['error']}")
            else:
                # Проверяем, сработал ли триггер на bias
                fairness_status = analysis.get("fairness_warning", {}).get("status", "")
                if "⚠️" in fairness_status:
                    print(f"✅ Готово (⚠️ Обнаружен Bias)")
                else:
                    print(f"✅ Готово")
                    
            report = {
                "candidate_id": candidate_id,
                "analysis": analysis
            }
            final_results.append(report)
            
        except Exception as e:
            print(f"❌ Ошибка обработки: {str(e)}")

        # Небольшая пауза, чтобы API не банило за спам (Rate Limits)
        if i < total_candidates:
            time.sleep(1.5)

    # Выводим сводную таблицу
    print_summary_table(final_results)

    # Сохраняем результаты
    output_file = 'final_results.json'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"💾 Результаты успешно сохранены в файл: {output_file}")
        print(f"⏱️  Общее время выполнения: {elapsed:.1f} сек.")
    except IOError as e:
        print(f"❌ Ошибка сохранения результатов: {e}")

if __name__ == "__main__":
    process_candidates()