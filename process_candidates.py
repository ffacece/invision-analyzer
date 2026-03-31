import json
import time
import sys
from include.main import CandidateAnalyzer

def _bar(score: float, width: int = 18) -> str:
    """Рендерит текстовый прогресс-бар для значения 0.0–1.0."""
    if not isinstance(score, (int, float)):
        return "?" * width
    filled = max(0, min(width, round(float(score) * width)))
    return "█" * filled + "░" * (width - filled)


def _safe_get(d: dict, *keys, default="N/A"):
    """Безопасно извлекает вложенное значение из словаря."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def print_explainability_cards(results: list):
    """Печатает детальную карточку explainability для каждого кандидата."""
    SEP = "─" * 74

    print(f"\n\n{'━' * 80}")
    print(f"  🔍 ДЕТАЛЬНЫЙ РАЗБОР КАНДИДАТОВ (EXPLAINABILITY)")
    print(f"{'━' * 80}")

    for r in results:
        cid = r["candidate_id"]
        a   = r["analysis"]

        print(f"\n  {SEP}")
        print(f"  📋 {cid}")
        print(f"  {SEP}")

        if "error" in a:
            print(f"  ⚠️  Анализ не выполнен: {a['error']}")
            continue

        s        = a.get("scores", {})
        bd       = a.get("score_breakdown", {})
        fi       = a.get("feature_impact", {})
        ev       = a.get("evidence", {})
        ai_l     = a.get("ai_risk_level", "Unknown")
        ai_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(ai_l, "⚪")

        lead = s.get("leadership") or 0.0
        moti = s.get("motivation") or 0.0
        grow = s.get("growth_path") or 0.0
        ai_r = s.get("ai_risk") or 0.0

        # ── Scores + bars + key factors ──────────────────────────────────
        print(f"  SCORES")
        for label, val, crit in [
            ("Leadership", lead, "leadership"),
            ("Motivation", moti, "motivation"),
            ("Growth    ", grow, "growth_path"),
        ]:
            bar = _bar(val)
            kf  = str(_safe_get(fi, crit, "key_factor"))[:32]
            print(f"    {label}  {val:.2f}  {bar}  → {kf}")
        print(f"    AI Risk    {ai_emoji} {ai_l} ({ai_r:.2f})")
        # ── Local text features ──────────────────────────────────────────────
        tf = a.get("text_features")
        if tf:
            print(f"\n  TEXT FEATURES  (локально, до LLM)")
            wc   = tf.get("word_count", "?")
            sc   = tf.get("sentence_count", "?")
            asl  = tf.get("avg_sentence_len", "?")
            ttr  = tf.get("type_token_ratio", "?")
            aimc = tf.get("ai_marker_count", "?")
            ivc  = tf.get("initiative_verb_count", "?")
            nc   = tf.get("number_count", "?")
            hf_ai= tf.get("hf_ai_score", "N/A")
            
            print(f"    Слов: {wc} | Предл: {sc} | Сред. длина: {asl} | TTR: {ttr}")
            print(f"    ИИ-маркеры: {aimc} | Глаголы действия: {ivc} | Числа: {nc}")
            if hf_ai != "N/A":
                print(f"    🤖 HF AI Score: {hf_ai} (HuggingFace Detector)")
        # ── Per-criterion reasoning ───────────────────────────────────────
        print(f"\n  REASONING")
        for label, crit in [
            ("Leadership", "leadership"),
            ("Motivation", "motivation"),
            ("Growth    ", "growth_path"),
        ]:
            text = str(_safe_get(bd, crit, "reasoning"))
            text = text[:65] + "..." if len(text) > 65 else text
            print(f"    {label}: {text}")

        # ── Feature weights ───────────────────────────────────────────────
        w_lead = _safe_get(fi, "leadership",  "weight_pct", default=0)
        w_moti = _safe_get(fi, "motivation",  "weight_pct", default=0)
        w_grow = _safe_get(fi, "growth_path", "weight_pct", default=0)
        if any(isinstance(w, (int, float)) and w > 0 for w in [w_lead, w_moti, w_grow]):
            print(f"\n  FACTOR WEIGHTS")
            print(f"    Leadership {w_lead}%  |  Motivation {w_moti}%  |  Growth {w_grow}%")

        # ── Evidence quotes ───────────────────────────────────────────────
        quotes = ev.get("highlighted_quotes", [])
        if quotes:
            print(f"\n  EVIDENCE")
            for q in quotes[:3]:
                q_s = str(q)[:68]
                print(f'    › "{q_s}"')

        # ── AI red flags ──────────────────────────────────────────────────
        flags = [f for f in ev.get("ai_red_flags", []) if f]
        if flags:
            print(f"\n  AI RED FLAGS")
            for fl in flags[:2]:
                print(f"    ⚑ {str(fl)[:68]}")

        # ── Summary ───────────────────────────────────────────────────────
        explanation = a.get("explanation", "")
        if explanation:
            print(f"\n  SUMMARY")
            words, lines, cur = explanation.split(), [], ""
            for w in words:
                if len(cur) + len(w) + 1 <= 70:
                    cur = f"{cur} {w}" if cur else w
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            for ln in lines[:4]:
                print(f"    {ln}")

        # ── Fairness warning ──────────────────────────────────────────────
        fw = a.get("fairness_warning", {})
        if "⚠️" in fw.get("status", ""):
            kw = fw.get("flagged_keywords", [])
            print(f"\n  ⚠️  BIAS DETECTED: {', '.join(kw)}")

    print(f"\n  {'━' * 74}\n")


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
    print(f"\n📊 Начинаем обработку {total_candidates} эссе через OpenRouter API...\n")

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

    # Выводим детальные explainability-карточки
    print_explainability_cards(final_results)

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