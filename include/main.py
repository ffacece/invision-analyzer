import hashlib
import time
import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
from transformers import models

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from natasha import (
        Segmenter, MorphVocab, NewsEmbedding,
        NewsNERTagger, Doc
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False

load_dotenv(override=True)

class CandidateAnalyzer:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API ключ не найден! Убедитесь, что файл .env существует "
                "и содержит OPENROUTER_API_KEY."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        self.models_priority = [
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ]

        # Кэш: md5(safe_text) → готовый результат анализа
        self._cache: dict = {}

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
        # HUGGINGFACE AI-DETECTION
        # ────────────────────────────────────────────
        if TRANSFORMERS_AVAILABLE:
            model_name = "AICodexLab/answerdotai-ModernBERT-base-ai-detector"
            print(f"⏳ Загрузка модели детекции ИИ ({model_name})...")
            try:
                hf_token = os.getenv("HF_TOKEN")
                self.ai_tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self.ai_model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
                self.ai_detector = pipeline("text-classification", model=self.ai_model, tokenizer=self.ai_tokenizer)
                print("✅ Модель детекции ИИ успешно загружена")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке модели детекции ИИ: {e}")
                self.ai_detector = None
        else:
            self.ai_detector = None
            print("⚠️ transformers не установлены. HF AI-детектор недоступен.")


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
            r'|абад[еуом]?|ополь|ополе|ополем)'
            r'\b'
        )

        # ────────────────────────────────────────────
        # УРОВЕНЬ 1: PROTECTED ATTRIBUTES (BIAS)
        # Удаляем всё, что может вызвать предвзятость
        # ────────────────────────────────────────────

        # Возраст: «мне 17 лет», «22-летний», «1998 года рождения»
        self.age_pattern = re.compile(
            r'\b(?:мне|мой возраст|возраст)\s*[-—]?\s*\d{1,2}\s*(?:лет|год[а-я]*)'
            r'|\b\d{1,2}[-\s]?лет(?:ний|няя|нее|них|нем|ней|нюю)?'
            r'|\b(?:родил(?:ся|ась)\s+в\s+)?\d{4}\s*(?:года?\s*рождения|г\.?\s*р\.?)'
            r'|\b(?:мне\s+)?\d{1,2}\b(?=\s*,?\s*(?:и я|учусь|работаю|живу))',
            re.IGNORECASE
        )

        # Пол/гендер: «я парень», «как девушка», «будучи мужчиной»
        self.gender_pattern = re.compile(
            r'\b(?:я\s+)?(?:парень|девушка|мужчина|женщина|юноша|девочка|мальчик)'
            r'|\b(?:как|будучи|являясь)\s+(?:парн[еюям]|девушк[еойи]|мужчин[аеойы]|женщин[аеойы])'
            r'|\b(?:мой|моя|моё)\s+(?:муж|жена|супруг[аи]?)',
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
                "aspect": "Сильный кандидат — конкретика и инициатива",
                "example": (
                    "Я самостоятельно изучил Python в ауле, организовал "
                    "IT-клуб для 15 подростков. Мы собрали дронов из запчастей "
                    "со свалки, чтобы мониторить экологию."
                ),
                "strength": (
                    "Конкретика, инициатива в условиях дефицита ресурсов, "
                    "отсутствие клише. Leadership ≥ 0.85, Growth ≥ 0.85."
                ),
            },
            {
                "aspect": "Средний кандидат — есть опыт, мало конкретики",
                "example": (
                    "Я участвовал в хакатоне и наша команда заняла 2-е место. "
                    "Это научило меня работать в команде. Хочу развиваться "
                    "в сфере технологий."
                ),
                "strength": (
                    "Есть реальный опыт, но без деталей личного вклада и масштаба. "
                    "Клише преобладают. Leadership 0.40–0.55, Growth 0.40–0.55."
                ),
            },
            {
                "aspect": "Слабый кандидат — декларации без фактов",
                "example": (
                    "Я считаю себя высокомотивированным индивидом с глубоким "
                    "пониманием технологических трендов. Моя цель — синергия "
                    "с инновационной средой. Уверен, что мой вклад будет неоценим."
                ),
                "strength": (
                    "Ноль конкретных фактов. Корпоративные клише без единого "
                    "реального действия. Leadership ≤ 0.20, Motivation ≤ 0.25."
                ),
            },
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
        # ОФЛАЙН-ПРИЗНАКИ ТЕКСТА
    # ════════════════════════════════════════════════

    def _extract_text_features(self, text: str) -> dict:
        """
        Вычисляет лингвистические признаки локально, до вызова LLM.
        Результат идёт в промпт как контекст и в итоговый JSON как
        объяснимое свидетельство.
        """
        words = re.findall(r'\b[а-яёa-zA-Z]+\b', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        word_count     = len(words)
        sentence_count = max(len(sentences), 1)
        avg_sent_len   = round(word_count / sentence_count, 1)
        ttr            = round(len(set(words)) / word_count, 3) if word_count else 0.0

        # Лексические маркеры ИИ-генерации
        AI_MARKERS = [
            'безусловно', 'важно отметить', 'в контексте', 'таким образом',
            'в заключение', 'стоит подчеркнуть', 'синергия', 'кроме того',
            'также стоит отметить', 'резюмируя', 'следует отметить',
            'в современном мире', 'трансформация', 'высокомотивированный',
            'неоценим', 'долгосрочной перспективе', 'реализации потенциала',
        ]
        ai_marker_count = sum(1 for m in AI_MARKERS if m in text.lower())

        # Глаголы конкретных действий (признак реального лидерства)
        INITIATIVE_VERBS = [
            'организовал', 'создал', 'запустил', 'основал', 'разработал',
            'провёл', 'провел', 'решил', 'открыл', 'собрал', 'написал',
            'придумал', 'нашёл', 'нашел', 'построил', 'инициировал',
            'реализовал', 'объединил', 'привлёк', 'привлек', 'обучил',
        ]
        initiative_verb_count = sum(
            1 for v in INITIATIVE_VERBS if v in text.lower()
        )

        # Числа как прокси конкретности высказываний
        number_count = len(re.findall(r'\b\d+\b', text))
        
        # Детекция ИИ с помощью HuggingFace (если доступно)
        hf_ai_score = None
        hf_ai_label = None
        if hasattr(self, 'ai_detector') and self.ai_detector is not None:
            # Ограничиваем длину текста для модели
            try:
                # В пайплайн передаем текст (до 8000 символов, чтобы влезть в лимит)
                hf_result = self.ai_detector(text[:8000])
                if hf_result and len(hf_result) > 0:
                    hf_ai_label = hf_result[0].get('label')
                    score = hf_result[0].get('score')
                    
                    # Приводим к формату (0.0 = человек, 1.0 = ИИ)
                    # Если модель возвращает 'LABEL_1' (ИИ) или 'LABEL_0' (человек) - адаптируйте в зависимости от выхода (ModernBERT: 1=ИИ)
                    if str(hf_ai_label).lower() in ['label_1', 'ai', '1', 'ai-generated']:
                        hf_ai_score = score
                    elif str(hf_ai_label).lower() in ['label_0', 'human', '0', 'human-written']:
                        hf_ai_score = 1.0 - score
                    else:
                        hf_ai_score = score # Фоллбэк
            except Exception as e:
                print(f"⚠️ Ошибка вызова HuggingFace детектора: {e}")

        return {
            "word_count":            word_count,
            "sentence_count":        sentence_count,
            "avg_sentence_len":      avg_sent_len,
            "type_token_ratio":      ttr,
            "ai_marker_count":       ai_marker_count,
            "initiative_verb_count": initiative_verb_count,
            "number_count":          number_count,
            "hf_ai_score":           round(hf_ai_score, 3) if hf_ai_score is not None else None,
            "hf_ai_label":           hf_ai_label,
        }

    # ════════════════════════════════════════════════
    # УРОВЕНЬ 3: ВАЛИДАЦИЯ ВЫХОДА НА BIAS
    # ════════════════════════════════════════════════

    @staticmethod
    def _validate_fairness(result: dict) -> dict:
        """
        Проверяет, что LLM не упомянула protected attributes
        в своём объяснении или evidence. Использует точный поиск по словам.
        """
        # Регулярное выражение для поиска точных слов (с учетом окончаний)
        bias_pattern = re.compile(
            r'\b(?:пол|гендер|мужчин[аеуы]|женщин[аеуы]|парень|парн[яю]|девушк[аеиу]|'
            r'возраст|молод(?:ой|ая|ые)|стар(?:ый|ая|ые)|юн(?:ый|ая|ые)|'
            r'религи[яию]|мусульман(?:ин|ка|е)|христиан(?:ин|ка|е)|атеист(?:ка|ы)?|'
            r'национальност[ьи]|этнич|каза[хш](?:ка|и)?|русск(?:ий|ая|ие)|узбе[кч](?:ка|и)?|'
            r'инвалид(?:ность|ы)?|овз|ограниченн|'
            r'малообеспеч|бедн|богат|нищ|'
            r'замужем|женат|разведен[аы]?|дет(?:ей|и|ям))\b',
            re.IGNORECASE
        )

        explanation = result.get("explanation", "")
        quotes = " ".join(result.get("evidence", {}).get("highlighted_quotes", []))
        check_text = f"{explanation} {quotes}"

        # Ищем все совпадения
        found_matches = bias_pattern.findall(check_text)
        
        # Оставляем только уникальные совпадения в нижнем регистре
        unique_flags = list(set([m.lower() for m in found_matches]))

        # Исключаем ложное срабатывание на слово "пол" если это часть "полгода" (страховка)
        if "пол" in unique_flags and "полгода" in check_text.lower():
            unique_flags.remove("пол")

        if unique_flags:
            result["fairness_warning"] = {
                "status": "⚠️ BIAS DETECTED IN OUTPUT",
                "flagged_keywords": unique_flags,
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
    # НОРМАЛИЗАЦИЯ ОТВЕТА LLM
    # ════════════════════════════════════════════════

    @staticmethod
    def _normalize_result(result: dict, text_features: dict) -> dict:
        """
        После парсинга JSON от LLM:
        - зажимает все scores в диапазон [0.0, 1.0]
        - гарантирует наличие score_breakdown и feature_impact для каждого критерия
        - прикрепляет локально вычисленные text_features
        - подменяет ai_risk на оценку от HuggingFace, если она доступна
        """
        scores = result.get("scores", {})
        
        # Интеграция HuggingFace детектора:
        if text_features.get("hf_ai_score") is not None:
            scores["ai_risk"] = text_features["hf_ai_score"]
            
            # Также обновляем ai_risk_level на основании этой оценки
            risk_val = scores["ai_risk"]
            if risk_val > 0.7:
                result["ai_risk_level"] = "High"
            elif risk_val > 0.3:
                result["ai_risk_level"] = "Medium"
            else:
                result["ai_risk_level"] = "Low"
                
        for key in ("leadership", "motivation", "growth_path", "ai_risk"):
            v = scores.get(key)
            if isinstance(v, (int, float)):
                scores[key] = round(max(0.0, min(1.0, float(v))), 3)

        bd = result.get("score_breakdown") or {}
        result["score_breakdown"] = bd
        fi = result.get("feature_impact") or {}
        result["feature_impact"] = fi
        for crit in ("leadership", "motivation", "growth_path"):
            if not isinstance(bd.get(crit), dict):
                bd[crit] = {"score": scores.get(crit), "reasoning": "N/A"}
            elif bd[crit].get("score") is None:
                bd[crit]["score"] = scores.get(crit)
            if not isinstance(fi.get(crit), dict):
                fi[crit] = {"key_factor": "N/A", "weight_pct": 0}

        result["text_features"] = text_features
        return result

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
            "score_breakdown": {
                "leadership": {"score": None, "reasoning": "N/A"},
                "motivation":  {"score": None, "reasoning": "N/A"},
                "growth_path": {"score": None, "reasoning": "N/A"},
            },
            "feature_impact": {
                "leadership": {"key_factor": "N/A", "weight_pct": 0},
                "motivation":  {"key_factor": "N/A", "weight_pct": 0},
                "growth_path": {"key_factor": "N/A", "weight_pct": 0},
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

        # Кэш — не тратим API-вызов на повторно встреченный текст
        cache_key = hashlib.md5(safe_text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Офлайн-признаки текста, вычисленные локально ДО отправки в LLM
        text_features = self._extract_text_features(safe_text)

        # УРОВЕНЬ 2: Промпт с явным запретом bias
        prompt = f"""
Ты — ведущий эксперт-лингвист и член приемной комиссии inVision U.
Проведи глубокий аудит эссе по двум направлениям:
  A) Детекция ИИ-генерации (ТЫ САМ ПРИНИМАЕШЬ РЕШЕНИЕ, сгенерирован текст или нет, опираясь исключительно на лингвистические маркеры)
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
Упоминание трудностей ценно ТОЛЬКО через призму ДЕЙСТВИЙ кандидата, а не через призму статуса.

═══════════════════════════════════════
ЗОЛОТЫЕ СТАНДАРТЫ (ЭТАЛОНЫ):
{json.dumps(self.golden_standards, ensure_ascii=False, indent=2)}

═══════════════════════════════════════
📊 АВТОМАТИЧЕСКИ ВЫЧИСЛЕННЫЕ ПРИЗНАКИ ТЕКСТА:
{json.dumps(text_features, ensure_ascii=False)}

Используй как объективный контекст (не для bias):
  • word_count < 80 → снижай confidence при детекции ИИ
  • ai_marker_count ≥ 4 → сильный сигнал ИИ-генерации
  • initiative_verb_count ≥ 3 → признак реальных действий (leadership ↑)
  • type_token_ratio < 0.45 → лексическая бедность (часто ИИ-паттерн)
  • number_count ≥ 3 → присутствует конкретика (specificity ↑)

═══════════════════════════════════════
A. ЧЕК-ЛИСТ ДЕТЕКЦИИ ИИ (8 КРИТЕРИЕВ)

Оцени каждый критерий от 1 до 10 (1 = живой человек, 10 = 100% ИИ).

1. ЛЕКСИЧЕСКОЕ РАЗНООБРАЗИЕ
   - Маркеры ИИ: «безусловно», «важно отметить», «в контексте», «таким образом», «давайте рассмотрим», «в заключение», «стоит подчеркнуть», «синергия»
   - Наличие разговорной лексики, живой речи, нестандартных оборотов

2. СТРУКТУРА И ФОРМАТИРОВАНИЕ
   - «Идеальная» структурированность
   - Шаблон «введение → тело → заключение» там, где он не нужен

3. СТИЛИСТИЧЕСКАЯ ОДНОРОДНОСТЬ
   - ИИ держит стабильный, ровный, корпоративный тон. Человек может допускать эмоциональные всплески.

4. ПЕРПЛЕКСИЯ И ПРЕДСКАЗУЕМОСТЬ
   - Банальность vs неожиданность мыслей. Общие слова без конкретики — признак генерации.

5. ФАКТИЧЕСКИЕ И ЛОГИЧЕСКИЕ ОСОБЕННОСТИ
   - Наличие конкретного, уникального личного опыта (ИИ часто избегает деталей и пишет абстрактно).

6. СИНТАКСИЧЕСКИЕ ПАТТЕРНЫ
   - Разнообразие конструкций предложений. Человек пишет более "рвано".
   - Наличие некоторого количества тире/дефисов. Если в предложении вместо дефиса можно поставить запятую, то скорее всего предложение написано ИИ

7. ПРАГМАТИКА И КОММУНИКАТИВНЫЕ МАРКЕРЫ
   - Чувствуется ли уникальный «голос» автора, позиция, страсть.

8. ПАТТЕРНЫ СВЯЗНОСТИ
   - Избыточные коннекторы ИИ («кроме того», «также стоит отметить») vs неформальные переходы.
   - Лишний англицизм и использование сленгов/юмора в несмешном виде. Если в предложении употребляются слова, которые в формальной речи не употребляют, то скорее всего предложение написано ИИ

ПРАВИЛА ДЕТЕКЦИИ:
- Короткие тексты (<100 слов) — снижай уверенность (confidence_percent) и укажи это в caveats.
- Эссе может быть гибридным (человек написал основу, ИИ "причесал").

═══════════════════════════════════════
B. ОЦЕНКА КАНДИДАТА (0.0 – 1.0)
Оценивай по принципу "Show, Don't Tell" (Покажи, а не расскажи). Обилие красивых корпоративных слов без реальных, подтверждающих действий, должно получать низкий балл лидерства.

- leadership: инициатива, способность вести за собой, брать ответственность в нестандартных ситуациях.
- motivation: глубина амбиций, искренность, масштаб мышления (влияние на общество).
- growth_path: способность действовать в условиях ограничений, самостоятельность, решение проблем.

Для каждого из трёх критериев ОБЯЗАТЕЛЬНО заполни поля объяснимости:
  • score_breakdown[criterion].reasoning — 1 конкретное предложение: ПОЧЕМУ именно такой балл.
    Ссылайся на конкретный факт или цитату из текста, а не на общие слова.
  • feature_impact[criterion].key_factor — решающий фактор в ≤8 словах
    (например: «организовал клуб без ресурсов», «только корпоративные клише»).
  • feature_impact[criterion].weight_pct — вес этого фактора в решении
    (число 0–100; сумма по трём критериям должна составлять ≈ 100).

═══════════════════════════════════════
ФОРМАТ — ТОЛЬКО валидный JSON, без Markdown и пояснений:

{{
  "scores": {{
    "leadership": 0.0,
    "motivation": 0.0,
    "growth_path": 0.0,
    "ai_risk": 0.0
  }},
  "score_breakdown": {{
    "leadership": {{ "score": 0.0, "reasoning": "1 предложение: конкретная причина балла" }},
    "motivation":  {{ "score": 0.0, "reasoning": "1 предложение: конкретная причина балла" }},
    "growth_path": {{ "score": 0.0, "reasoning": "1 предложение: конкретная причина балла" }}
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
    "leadership": {{ "key_factor": "Фактор в ≤8 словах", "weight_pct": 0 }},
    "motivation":  {{ "key_factor": "Фактор в ≤8 словах", "weight_pct": 0 }},
    "growth_path": {{ "key_factor": "Фактор в ≤8 словах", "weight_pct": 0 }}
  }},
  "evidence": {{
    "highlighted_quotes": ["Цитата 1", "Цитата 2"],
    "ai_red_flags": ["Красный флаг 1 (если есть)"]
  }},
  "explanation": "Краткое, но предельно конкретное обоснование выставленных баллов (до 3 предложений). Опирайся на факты из текста.",
  "ai_risk_level": "Low/Medium/High"
}}

═══════════════════════════════════════
ТЕКСТ ДЛЯ АНАЛИЗА:
\"\"\"{safe_text}\"\"\"
"""

        errors_log = []

        for model_id in self.models_priority:
            for attempt in range(3):
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
                        break  # к следующей модели

                    result["model_used"] = model_id
                    result["privacy_applied"] = True

                    # Нормализация схемы + прикрепление text_features
                    result = self._normalize_result(result, text_features)

                    # УРОВЕНЬ 3: Валидация выхода на bias
                    result = self._validate_fairness(result)

                    # Сохраняем в кэш перед возвратом
                    self._cache[cache_key] = result
                    return result

                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = (
                        "429" in err_str
                        or "rate_limit" in err_str.lower()
                        or "rate limit" in err_str.lower()
                    )
                    if is_rate_limit and attempt < 2:
                        time.sleep(2 ** attempt)  # 1s, 2s
                        continue
                    errors_log.append(f"Ошибка на {model_id}: {err_str}")
                    break  # к следующей модели

        return self._fallback_result(
            "; ".join(errors_log) if errors_log else "Все модели недоступны."
        )