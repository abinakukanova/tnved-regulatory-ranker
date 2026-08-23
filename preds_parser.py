import argparse
import json
import pandas as pd
from pathlib import Path


def load_declarations(data_dir):
    """Загружает декларации из declarations.jsonl и возвращает dict {id: record}."""
    decl_path = Path(data_dir) / 'declarations.jsonl'
    declarations = {}
    with open(decl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                declarations[rec['declaration_id']] = rec
    return declarations


def load_regulations(data_dir):
    """Загружает регуляции из regulations.jsonl и возвращает dict {id: record}."""
    reg_path = Path(data_dir) / 'regulations.jsonl'
    regulations = {}
    with open(reg_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                regulations[rec['regulation_id']] = rec
    return regulations


def get_declaration_text(decl):
    """Формирует текст декларации из доступных полей."""
    parts = []
    if 'G31_1' in decl and decl['G31_1']:
        parts.append(decl['G31_1'])
    if 'desc_extention' in decl and decl['desc_extention']:
        parts.append(decl['desc_extention'])
    return ' '.join(parts) if parts else ''


def get_regulation_text(reg):
    """Формирует текст регуляции из доступных полей."""
    parts = []
    if 'code' in reg and reg['code']:
        parts.append(f"Код: {reg['code']}")
    if 'description' in reg and reg['description']:
        parts.append(reg['description'])
    if 'notes' in reg and reg['notes']:
        parts.append(f"Примечания: {reg['notes'][:200]}...")  # усекаем для краткости
    if 'explanation' in reg and reg['explanation']:
        parts.append(f"Пояснение: {reg['explanation']}")
    return ' '.join(parts) if parts else ''


def main():
    parser = argparse.ArgumentParser(description='Создание Excel-файла для анализа предсказаний')
    parser.add_argument('--predictions', type=str, default='./out/predictions.csv',
                        help='Путь к predictions.csv')
    parser.add_argument('--data', type=str, default='./data',
                        help='Путь к папке с данными (declarations.jsonl, regulations.jsonl)')
    parser.add_argument('--output', type=str, default='./analysis.xlsx',
                        help='Путь для сохранения Excel-файла')
    args = parser.parse_args()

    # Загружаем данные
    print("Загрузка деклараций...")
    declarations = load_declarations(args.data)
    print(f"Загружено {len(declarations)} деклараций")

    print("Загрузка регуляций...")
    regulations = load_regulations(args.data)
    print(f"Загружено {len(regulations)} регуляций")

    # Читаем predictions.csv
    print("Чтение predictions.csv...")
    pred_df = pd.read_csv(args.predictions)

    # Проверяем наличие всех необходимых ID
    missing_decl = set(pred_df['declaration_id']) - set(declarations.keys())
    missing_reg = set(pred_df['regulation_id']) - set(regulations.keys())
    if missing_decl:
        print(f"Предупреждение: нет данных для деклараций: {missing_decl}")
    if missing_reg:
        print(f"Предупреждение: нет данных для регуляций: {missing_reg}")


    print("Добавление текстов...")
    pred_df['declaration_text'] = pred_df['declaration_id'].apply(
        lambda x: get_declaration_text(declarations.get(x, {}))
    )
    pred_df['regulation_text'] = pred_df['regulation_id'].apply(
        lambda x: get_regulation_text(regulations.get(x, {}))
    )


    pred_df = pred_df.sort_values(['declaration_id', 'rank'])


    pred_df['regulation_full_text'] = pred_df['regulation_id'].apply(
        lambda x: ' '.join([
            regulations.get(x, {}).get('description', ''),
            regulations.get(x, {}).get('notes', ''),
            regulations.get(x, {}).get('explanation', '')
        ])
    )


    cols = [
        'declaration_id', 'rank', 'regulation_id', 'score',
        'declaration_text', 'regulation_text', 'regulation_full_text'
    ]
    pred_df = pred_df[cols]


    print(f"Сохранение в {args.output}...")
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        pred_df.to_excel(writer, sheet_name='Предсказания', index=False)
        stats = pred_df.groupby('declaration_id').agg(
            mean_score=('score', 'mean'),
            max_score=('score', 'max'),
            min_score=('score', 'min')
        ).reset_index()
        stats.to_excel(writer, sheet_name='Статистика', index=False)

    print("Готово!")


if __name__ == '__main__':
    main()