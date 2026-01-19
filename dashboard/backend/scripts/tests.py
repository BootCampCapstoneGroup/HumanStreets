import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
# wav_french = multilingual_model.generate(french_text, language_id="fr")
# ta.save("test-french.wav", wav_french, model.sr)

# arabic
arabic_text = "مرحبا، كيف حالك؟ هذا هو نموذج تحويل النص إلى كلام متعدد اللغات Chatterbox، وهو يدعم 23 لغة."
wav_arabic = multilingual_model.generate(arabic_text, language_id="ar")
ta.save("test-arabic.wav", wav_arabic, model.sr)



# #chiness 
# chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
# wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
# ta.save("test-chinese.wav", wav_chinese, model.sr)