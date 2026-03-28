import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

try:
    from natasha import (
        Segmenter, MorphVocab, NewsEmbedding,
        NewsNERTagger, Doc
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False

load_dotenv()


class CandidateAnalyzer:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "API ключ не найден! Убедитесь, что файл .env существует "
                "и содержит GROQ_API_KEY."
            )

        self.client = Groq(api_key=api_key)

        self.models_priority = [
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
        ]

        # ────────────────────────────────────────────
        # NATASHA NER
        # ────────────────────────────────────────────
        if NATASHA_AVAILABLE:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            self.ner_tagger = NewsNERTagger(emb)
            print("✅ natasha загружена — NER-анонимизация активна")
        else:
            print(
                "⚠️ natasha не установлена (pip install natasha). "
                "Regex-фолбэк активен."
            )

        # ────────────────────────────────────────────
        # REGEX PATTERNS
        # ────────────────────────────────────────────

        # Имена с дефисами и отчествами
        self.name_pattern = re.compile(
            r'\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?'
            r'\s+'
            r'[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?'
            r'(?:\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)?'
            r'\b'
        )

        # Контекстные геомаркеры
        self.geo_context_pattern = re.compile(
            r'(?:из|в|на|под|около|возле|близ|'
            r'г\.|гор\.|город[еау]?|'
            r'сел[оауе]|аул[еау]?|'
            r'посел[оа]?к[еау]?|пос\.|'
            r'район[еау]?|р-н[еау]?|'
            r'област[иья]|обл\.|'
            r'улиц[аеыу]|ул\.|'
            r'проспект[еау]?|пр\.)'
            r'\s+'
            r'([А-ЯЁ][а-яё]+(?:[- ][А-ЯЁа-яё]+)*)',
            re.IGNORECASE
        )

        # Морфологические суффиксы городов
        self.geo_suffix_pattern = re.compile(
            r'\b[А-ЯЁ][а-яё]+'
            r'(?:ск[еуом]?|град[еуом]?|бург[еуом]?|горск[еуом]?'
            r'|абад[еуом]?|��поль|ополе|ополем)'
            r'\b'
        )

        # ────────────────────────────────────────────
        # УРОВЕНЬ 1: PROTECTED ATTRIBUTES (BIAS)
        # Удаляем всё, что может вызвать предвзятость
        # ────────────────────────────────────────────

        # Возраст: «мне 17 лет», «22-летний», «1998 года рождения»
        self.age_pattern = re.compile(
            r'\b(?:мне|мой возраст|возраст)\s*[-—]?\s*\d{1,2}\s*(?:лет|год[а-я]*)'
            r'|\b\d{1,2}[-\s]?л��т(?:ний|няя|нее|них|нем|ней|нюю)?'
            r'|\b(?:родил(?:ся|ась)\s+в\s+)?\d{4}\s*(?:года?\s*рождения|г\.?\s*р\.?)'
            r'|\b(?:мне\s+)?\d{1,2}\b(?=\s*,?\s*(?:и я|учусь|работаю|живу))',
            re.IGNORECASE
        )

        # Пол/гендер: «я парень», «как девушка», «будучи мужчиной»
        self.gender_pattern = re.compile(
            r'\b(?:я\s+)?(?:парень|девушка|мужчина|женщина|юноша|девочка|мальчик)'
            r'|\b(?:как|будучи|являясь)\s+(?:парн[еюям]|девушк[еойи]|мужчин[аеойы]|женщин[аеойы])'
            r'|\b(?:мой|моя|��оё)\s+(?:муж|жена|супруг[аи]?)',
            re.IGNORECASE
        )

        # Религия: «мусульманин», «христианка», «намаз», «церковь» (в контексте принадлежности)
        self.religion_pattern = re.compile(
            r'\b(?:я\s+)?(?:мусульман(?:ин|ка)|христиан(?:ин|ка)|буддист(?:ка)?'
            r'|атеист(?:ка)?|иуде[йя]|православн[аыйое]+|католи[кч])'
            r'|\b(?:хожу\s+в\s+(?:мечеть|церковь|синагогу|храм))'
            r'|\b(?:читаю\s+(?:намаз|молитв[уы]))',
            re.IGNORECASE
        )

        # Этническая принадлежность: «я казах», «по национальности узбечка»
        self.ethnicity_pattern = re.compile(
            r'\b(?:я\s+)?(?:каза[хш](?:ка)?|русск(?:ий|ая)|узбе[кч](?:ка)?'
            r'|кирги[зс](?:ка)?|таджи[кч](?:ка)?|уйгур(?:ка)?'
            r'|татар(?:ин|ка)?|корее[цч]|немец|немка|чечен(?:ец|ка)?'
            r'|армян(?:ин|ка)?|грузин(?:ка)?|азербайджан(?:ец|ка)?)'
            r'|\b(?:по\s+национальности|по\s+происхождению)\s+\w+',
            re.IGNORECASE
        )

        # Инвалидность / состояние здоровья (в контексте самоидентификации)
        self.disability_pattern = re.compile(
            r'\b(?:у\s+меня\s+)?(?:инвалидность|ОВЗ|особ(?:ые|ых)\s+(?:потребност|нужд))'
            r'|\b(?:я\s+)?(?:незряч|слабовидящ|глух(?:ой|ая)|слабослыш)'
            r'|\b(?:передвигаюсь\s+на\s+(?:коляске|костылях))',
            re.IGNORECASE
        )

        # Семейное положение / дети
        self.family_pattern = re.compile(
            r'\b(?:я\s+)?(?:жена[тт]|замужем|разведен[аы]?|холост|не\s+замужем)'
            r'|\b(?:у\s+меня\s+)?\d+\s+(?:дет(?:ей|и)|ребен(?:ок|ка))',
            re.IGNORECASE
        )

        # Социально-экономический статус: «малообеспеченная семья», «неполная семья»
        self.socioeconomic_pattern = re.compile(
            r'\b(?:малообеспеченн|многодетн|неполн|малоимущ|бедн|нищ)'
            r'(?:[аоыейую]+\s+)?(?:семь[яиею]|родител[иьяей])'
            r'|\b(?:детдом|детский\s+дом|сирот[аы]|приёмн(?:ая|ый))',
            re.IGNORECASE
        )

        # Все bias-паттерны в одном списке
        self.bias_patterns = [
            (self.age_pattern, "[ВОЗРАСТ_УДАЛЁН]"),
            (self.gender_pattern, "[ПОЛ_УДАЛЁН]"),
            (self.religion_pattern, "[РЕЛИГИЯ_УДАЛЕНА]"),
            (self.ethnicity_pattern, "[ЭТНИЧНОСТЬ_УДАЛЕНА]"),
            (self.disability_pattern, "[ЗДОРОВЬЕ_УДАЛЕНО]"),
            (self.family_pattern, "[СЕМЬЯ_УДАЛЕНО]"),
            (self.socioeconomic_pattern, "[СОЦ_СТАТУС_УДАЛЁН]"),
        ]

        # ────────────────────────────────────────────
        # ЗОЛОТЫЕ СТАНДАРТЫ
        # ────────────────────────────────────────────
        self.golden_standards = [
            {
                "aspect": "Идеальный пройденный путь и лидерство",
                "example": (
                    "Я самостоятельно изучил Python в ауле, организовал "
                    "IT-клуб для 15 подростков. Мы собрали дронов из запчастей "
                    "со свалки, чтобы мониторить экологию."
                ),
                "strength": (
                    "Конкретика, инициатива в условиях дефицита ресурсов, "
                    "отсутствие клише."
                ),
            }
        ]

    # ════════════════════════════════════════════════
    # УРОВЕНЬ 1: ПОЛНАЯ АНОНИМИЗАЦИЯ
    # ════════════════════════════════════════════════

    def anonymize_text(self, text: str) -> str:
        """
        Трёхэтапная анонимизация:
        1. NER / Regex — имена, локации, организации
        2. Bias removal — возраст, пол, религия, этничность и т.д.
        3. Финальная чистка
        """
        if not text:
            return ""

        # Шаг 1: Имена и локации
        if NATASHA_AVAILABLE:
            text = self._anonymize_with_natasha(text)
        else:
            text = self._anonymize_with_regex(text)

        # Шаг 2: Protected attributes (bias removal)
        text = self._remove_protected_attributes(text)

        return text

    def _anonymize_with_natasha(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        replacements = {
            "PER": "[КАНДИДАТ]",
            "LOC": "[ЛОКАЦИЯ]",
            "ORG": "[ОРГАНИЗАЦИЯ]",
        }

        spans = sorted(doc.spans, key=lambda s: s.start, reverse=True)
        result = text
        for span in spans:
            placeholder = replacements.get(span.type)
            if placeholder:
                result = result[:span.start] + placeholder + result[span.stop:]
        return result

    def _anonymize_with_regex(self, text: str) -> str:
        text = self.name_pattern.sub("[КАНДИДАТ]", text)
        text = self.geo_context_pattern.sub(
            lambda m: m.group(0).replace(m.group(1), "[ЛОКАЦИЯ]"), text
        )
        text = self.geo_suffix_pattern.sub("[ЛОКАЦИЯ]", text)
        return text

    def _remove_protected_attributes(self, text: str) -> str:
        """Удаляет все protected attributes, которые могут вызвать bias."""
        for pattern, replacement in self.bias_patterns:
            text = pattern.sub(replacement, text)
        return text

    # ════════════════════════════════════════════════
    # УРОВЕНЬ 3: ВАЛИДАЦИЯ ВЫХОДА НА BIAS
    # ════════════════════════════════════════════════

    @staticmethod
    def _validate_fairness(result: dict) -> dict:
        """
        Проверяет, что LLM не упомянула protected attributes
        в своём объяснении или evidence.
        """
        bias_keywords = [
            "пол", "гендер", "мужчина", "женщина", "парень", "девушка",
            "возраст", "молодой", "старый", "юный",
            "религи", "мусульман", "христиан", "атеист",
            "национальност", "этнич", "казах", "русск", "узбек",
            "инвалид", "овз", "ограниченн",
            "малообеспеч", "бедн", "богат", "нищ",
            "замужем", "женат", "разведен", "детей",
        ]

        explanation = result.get("explanation", "").lower()
        quotes = " ".join(result.get("evidence", {}).get("highlighted_quotes", [])).lower()
        check_text = explanation + " " + quotes

        bias_found = [kw for kw in bias_keywords if kw in check_text]

        if bias_found:
            result["fairness_warning"] = {
                "status": "⚠️ BIAS DETECTED IN OUTPUT",
                "flagged_keywords": bias_found,
                "action": (
                    "Обнаружены упоминания protected attributes в ответе LLM. "
                    "Рекомендуется ручная проверка перед использованием."
                ),
            }
        else:
            result["fairness_warning"] = {
                "status": "✅ CLEAN",
                "flagged_keywords": [],
                "action": "Bias не обнаружен в выходных данных.",
            }

        return result

    # ════════════════════════════════════════════════
    # БЕЗОПАСНЫЙ ПАРСИНГ JSON
    # ════════════════════════════════════════════════

    @staticmethod
    def _safe_parse_json(raw: str) -> dict | None:
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        md_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL
        )
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # ════════════════════════════════════════════════
    # ЗАГЛУШКА
    # ════════════════════════════════════════════════

    @staticmethod
    def _fallback_result(reason: str) -> dict:
        return {
            "scores": {
                "leadership": None,
                "motivation": None,
                "growth_path": None,
                "ai_risk": None,
            },
            "ai_detection": {
                "criteria_scores": {
                    "lexical_diversity": None,
                    "structure": None,
                    "stylistic_uniformity": None,
                    "predictability": None,
                    "factual_logic": None,
                    "syntax": None,
                    "pragmatics": None,
                    "coherence_patterns": None,
                },
                "mean_score": None,
                "confidence_percent": 0,
                "caveats": ["Анализ не был выполнен"],
            },
            "feature_impact": {
                "main_positive_factor": "N/A",
                "factor_weight": 0,
            },
            "evidence": {
                "highlighted_quotes": [],
                "ai_red_flags": [],
            },
            "explanation": f"Анализ не выполнен: {reason}",
            "ai_risk_level": "Unknown",
            "model_used": "none",
            "privacy_applied": True,
            "fairness_warning": {
                "status": "N/A",
                "flagged_keywords": [],
                "action": "Анализ не выполнен",
            },
            "error": reason,
        }

    # ════════════════════════════════════════════════
    # ОСНОВНОЙ МЕТОД
    # ════════════════════════════════════════════════

    def analyze_essay(self, raw_essay_text: str) -> dict:
        # УРОВЕНЬ 1: Анонимизация + удаление bias ДО отправки в LLM
        safe_text = self.anonymize_text(raw_essay_text)

        # УРОВЕНЬ 2: Промпт с явным запретом bias
        prompt = f"""
Ты — ведущий эксперт-лингвист и член приемной комиссии inVision U.
Проведи аудит эссе по двум направлениям:
  A) Детекция ИИ-генерации
  B) Оценка лидерского потенциала кандидата

═══════════════════════════════════════
⛔ КРИТИЧЕСКИ ВАЖНО — ПРАВИЛА FAIRNESS:

Ты ОБЯЗАН оценивать ТОЛЬКО:
  ✅ Лидерские качества и инициативу
  ✅ Глубину мотивации
  ✅ Пройденный путь и преодолённые трудности
  ✅ Конкретику действий и результатов
  ✅ Подлинность текста (детекция ИИ)

Ты НЕ ИМЕЕШЬ ПРАВА учитывать при оценке:
  ❌ Пол, гендер, сексуальную ориентацию
  ❌ Возраст
  ❌ Национальность, этническую принадлежность, расу
  ❌ Религию или отсутствие религии
  ❌ Инвалидность, состояние здоровья
  ❌ Семейное положение, наличие/отсутствие детей
  ❌ Социально-экономический статус семьи
  ❌ Географическое происхождение (город vs село)
  ❌ Тип школы (частная, государственная, сельская)

Если в тексте остались такие маркеры — ПОЛНОСТЬЮ ИГНОРИРУЙ их.
Они НЕ ДОЛЖНЫ влиять на оценку ни в положительную, ни в отрицательную сторону.
Упоминание трудностей ценно ТОЛЬКО через призму действий кандидата, а не через призму статуса.

═══════════════════════════════════════
ЗОЛОТЫЕ СТАНДАРТЫ (ЭТАЛОНЫ):
{json.dumps(self.golden_standards, ensure_ascii=False, indent=2)}

═══════════════════════════════════════
A. ЧЕК-ЛИСТ ДЕТЕКЦИИ ИИ (8 КРИТЕРИЕВ)

Оцени каждый критерий от 1 до 10 (1 = типично для человека, 10 = типично для ИИ).

1. ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ
   - Маркеры ИИ: «безусловно», «важно отметить», «в контексте», «таким образом», «давайте рассмотрим», «в заключение», «стоит подчеркнуть», «синергия»
   - Разговорная лексика, сленг, жаргон, диалектизмы
   - Повторы vs нестандартные слова

2. СТРУКТУРА И ФОРМАТИРОВАНИЕ
   - «Идеальная» структурированность
   - Одинаковая длина абзацев
   - Шаблон «введение → тело → заключение» без необходимости

3. СТИЛИСТИЧЕСКАЯ ОДНОРОДНОСТЬ
   - Стабильный vs «плавающий» тон
   - Эмоциональные «всплески» / «провалы»

4. ПЕРПЛЕКСИЯ И ПРЕДСКАЗУЕМОСТЬ
   - Банальность vs неожиданность фраз
   - Нелинейная логика
   - Клише и «безопасные» формулировки

5. ФАКТИЧЕСКИЕ И ЛОГИЧЕСКИЕ ОСОБЕННОСТИ
   - Фактические неточности ИИ
   - Хеджирование
   - Личный опыт, детали, уникальные примеры

6. СИНТАКСИЧЕСКИЕ ПАТТЕРНЫ
   - Разнообразие конструкций
   - Ошибки, опечатки, незаконченные мысли
   - Длинные сложноподчинённые предложения

7. ПРАГМАТИКА И КОММУНИКАТИВНЫЕ МАРКЕРЫ
   - «Голос» автора, позиция
   - Ирония, сарказм, юмор
   - Обращение к читателю

8. ПАТТЕРНЫ СВЯЗНОСТИ
   - Избыточные коннекторы ИИ
   - Неформальные переходы человека

ПРАВИЛА ДЕТЕКЦИИ:
- Совокупность признаков, не один
- Человек мог редактировать ИИ-текст или писать «гладко»
- Короткие тексты (<100 слов) — снижай уверенность
- Грамотность ≠ искусственность

═══════════════════════════════════════
B. ОЦЕНКА КАНДИДАТА (0.0 – 1.0)

- leadership: инициатива, влияние, организация действий
- motivation: глубина, искренность, внутренний драйв
- growth_path: траектория роста, преодоление, конкретные шаги

═══════════════════════════════════════
ФОРМАТ — ТОЛЬКО валидный JSON, без Markdown:

{{
  "scores": {{
    "leadership": 0.0,
    "motivation": 0.0,
    "growth_path": 0.0,
    "ai_risk": 0.0
  }},
  "ai_detection": {{
    "criteria_scores": {{
      "lexical_diversity": 0,
      "structure": 0,
      "stylistic_uniformity": 0,
      "predictability": 0,
      "factual_logic": 0,
      "syntax": 0,
      "pragmatics": 0,
      "coherence_patterns": 0
    }},
    "mean_score": 0.0,
    "confidence_percent": 0,
    "caveats": []
  }},
  "feature_impact": {{
    "main_positive_factor": "",
    "factor_weight": 0
  }},
  "evidence": {{
    "highlighted_quotes": [],
    "ai_red_flags": []
  }},
  "explanation": "",
  "ai_risk_level": "Low/Medium/High"
}}

═══════════════════════════════════════
ТЕКСТ ДЛЯ АНАЛИЗА:
\"\"\"{safe_text}\"\"\"
"""

        errors_log = []

        for model_id in self.models_priority:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fair, unbiased evaluator. "
                                "You judge candidates ONLY by their actions, "
                                "initiative, and growth — NEVER by demographics. "
                                "Output ONLY valid JSON. No markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=model_id,
                    temperature=0.15,
                    response_format={"type": "json_object"},
                )

                raw_res = chat_completion.choices[0].message.content
                result = self._safe_parse_json(raw_res)

                if result is None:
                    msg = (
                        f"Модель {model_id} вернула невалидный JSON. "
                        f"Первые 200 символов: {raw_res[:200]}"
                    )
                    print(f"⚠️ {msg}")
                    errors_log.append(msg)
                    continue

                result["model_used"] = model_id
                result["privacy_applied"] = True

                # УРОВЕНЬ 3: Валидация выхода на bias
                result = self._validate_fairness(result)

                return result

            except Exception as e:
                msg = f"Ошибка на {model_id}: {str(e)}"
                print(f"⚠️ {msg}")
                errors_log.append(msg)
                continue

        return self._fallback_result(
            "; ".join(errors_log) if errors_log else "Все модели недоступны."
        )