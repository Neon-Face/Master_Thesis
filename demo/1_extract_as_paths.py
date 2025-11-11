import argparse
from mrtparse import Reader, BGP_ATTR_T, AS_PATH_SEG_T, MRT_T, BGP_MSG_T

def extract_as_paths(mrt_file_path, output_file_path):
    """
    Reads an MRT file and extracts all AS_PATHs from BGP ANNOUNCE messages,
    writing them to a text file, one per line.
    """
    print(f"Starting to process MRT file: {mrt_file_path}")
    
    path_count = 0
    record_count = 0
    with open(output_file_path, 'w') as f_out:
        # Use mrtparse.Reader to iterate through all records in the MRT file
        for m in Reader(mrt_file_path):
            record_count += 1
            # Add a try-except block to gracefully handle malformed records
            try:
                # We only care about BGP4MP messages
                # CORRECTED LINE: Cast to list before accessing element
                if list(m.data['type'])[0] != MRT_T['BGP4MP']:
                    continue

                # We only care about BGP UPDATE messages
                bgp_message = m.data.get('bgp_message', {})
                # CORRECTED LINE: Cast to list before accessing element
                print(bgp_message)
                if not bgp_message or list(bgp_message.get('type', [0]))[0] != BGP_MSG_T['UPDATE']:
                    continue
                
                # Find the AS_PATH attribute in the message
                as_path = None
                for attr in bgp_message.get('path_attributes', []):
                    # CORRECTED LINE: Cast to list before accessing element
                    if list(attr['type'])[0] == BGP_ATTR_T['AS_PATH']: 
                        as_path = attr['value']
                        break
                
                if not as_path:
                    continue

                # --- Process the AS_PATH and write it to the file ---
                path_str_list = []
                for seg in as_path:
                    # We are only interested in simple sequences for this project
                    if list(seg['type'])[0] == AS_PATH_SEG_T['AS_SEQUENCE']: 
                        path_str_list.extend(seg['value'])

                if path_str_list:
                    # Ensure the origin AS (last one) is present
                    if len(path_str_list) > 0:
                        f_out.write(' '.join(path_str_list) + '\n')
                        path_count += 1

                        if path_count % 100000 == 0:
                            print(f" ... processed {record_count} records, found {path_count} AS_PATHs")
            
            except (KeyError, IndexError) as e:
                # If a record is missing a field or has an unexpected structure, skip it.
                # print(f"Skipping malformed record {record_count}: {e}") # Uncomment for debugging
                continue

    print(f"\nDone! Processed {record_count} total records.")
    print(f"Extracted {path_count} valid AS_PATHs to '{output_file_path}'.")

def main():
    parser = argparse.ArgumentParser(
        description='Extracts AS_PATHs from an MRT file to a text corpus.'
    )
    parser.add_argument(
        'mrt_file', 
        help='Path to the input MRT file (e.g., updates.20251014.0000)'
    )
    parser.add_argument(
        '-o', '--output', 
        default='data/as_paths_corpus.txt', 
        help='Path to the output text file (default: as_paths_corpus.txt)'
    )
    args = parser.parse_args()

    extract_as_paths(args.mrt_file, args.output)

if __name__ == '__main__':
    main()