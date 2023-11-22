import json


def normalize(data):
    chinese_punctuations = " ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
    english_punctuations = " !?.\"#$%&\'()*+,-\:;<>=@[]\\^_`{}|~"
    data = data.strip(chinese_punctuations)
    data = data.strip(english_punctuations)
    data = data.lower()
    return data


def load_data(data_file):
    with open(data_file) as f:
        lines = f.readlines()
    refs = []
    preds = []
    extras = []
    for line in lines:
        info = json.loads(line)
        reference = info["references"]
        prediction = info["predictions"]
        info.pop("references")
        info.pop("predictions")
        reference = [normalize(ref) for ref in reference]
        prediction = [normalize(pred) for pred in prediction]
        refs.append(reference)
        preds.append(prediction)
        extras.append(info)  # store remaining information 
    return preds, refs, extras

