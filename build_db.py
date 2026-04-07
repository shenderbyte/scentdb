import pandas as pd
import json

df = pd.read_csv('fragrances.csv')

print(f"Total fragrances: {len(df)}")

df = df.dropna(subset=['Perfume', 'Brand'])

def build_text(row):
    parts = []
    parts.append(f"{row['Perfume']} by {row['Brand']}")
    
    if pd.notna(row['Top']):
        parts.append(f"Top notes: {row['Top']}")
    if pd.notna(row['Middle']):
        parts.append(f"Middle notes: {row['Middle']}")
    if pd.notna(row['Base']):
        parts.append(f"Base notes: {row['Base']}")
    
    accords = [row[f'mainaccord{i}'] for i in range(1,6) if pd.notna(row[f'mainaccord{i}'])]
    if accords:
        parts.append(f"Accords: {', '.join(accords)}")
    
    if pd.notna(row['Gender']):
        parts.append(f"Gender: {row['Gender']}")
    if pd.notna(row['Country']):
        parts.append(f"Country: {row['Country']}")
        
    return '. '.join(parts)

df['text'] = df.apply(build_text, axis=1)

print("\nExample text for first fragrance:")
print(df['text'].iloc[0])

df.to_json('fragrances_processed.json', orient='records', indent=2)
print(f"\nSaved {len(df)} fragrances to fragrances_processed.json")