import os
import csv
import json

EXCLUDED_SESSIONS = {"342", "394", "398", "460"}
EXTRACTED_PATH = '../data/extracted'
PROCESSED_PATH = '../data/processed'

def convert_transcript_to_json_participant_only(input_csv, output_json):
    data = []
    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            speaker = row['speaker'].strip().lower()
            if speaker == 'participant':
                data.append({
                    "start_time": float(row['start_time']),
                    "stop_time": float(row['stop_time']),
                    "value": row['value'].strip()
                })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file, indent=4)


def batch_process_all_transcripts_participant_only():
    for session_folder in os.listdir(EXTRACTED_PATH):
        session_path = os.path.join(EXTRACTED_PATH, session_folder)
        session_id = session_folder.split('_')[0]  # e.g., "300" from "300_P"

        if session_id in EXCLUDED_SESSIONS:
            print(f"Skipping excluded session: {session_folder}")
            continue

        transcript_csv = os.path.join(session_path, f"{session_id}_TRANSCRIPT.csv")
        if os.path.exists(transcript_csv):
            output_folder = os.path.join(PROCESSED_PATH, session_folder)
            output_json = os.path.join(output_folder, f"{session_id}_PARTICIPANT_TRANSCRIPT.json")
            print(f"Processing participant speech only for session: {session_id}")
            convert_transcript_to_json_participant_only(transcript_csv, output_json)
        else:
            print(f"TRANSCRIPT.csv not found for {session_id}, skipping...")


if __name__ == "__main__":
    batch_process_all_transcripts_participant_only()
