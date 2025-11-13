import argparse
import csv
from datetime import datetime
from mrtparse import Reader, MRT_T, BGP_MSG_T, BGP_ATTR_T, AS_PATH_SEG_T, TD_V2_ST, BGP4MP_ST, ORIGIN_T

PEER_INDEX_TABLE = {}

def extract_bgp_data(mrt_file_path, output_file_path):
    """
    Reads an MRT file and extracts key BGP attributes from each announcement,
    writing them to a CSV file.
    """
    print(f"Starting to process MRT file: {mrt_file_path}")
    
    row_count = 0
    record_count = 0

    with open(output_file_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        
        header = [
            'timestamp', 'event_type', 'peer_ip', 'peer_as', 
            'prefix', 'as_path', 'origin', 'next_hop'
        ]
        writer.writerow(header)

        for m in Reader(mrt_file_path):
            record_count += 1
            bgp_data = {}

            try:
                mrt_type_code = list(m.data['type'])[0]
                timestamp_val = list(m.data['timestamp'])[0]
                bgp_data['timestamp'] = datetime.utcfromtimestamp(timestamp_val).isoformat()

                if mrt_type_code == MRT_T['TABLE_DUMP_V2']:
                    subtype_code = list(m.data['subtype'])[0]
                    if subtype_code == TD_V2_ST['PEER_INDEX_TABLE']:
                        global PEER_INDEX_TABLE
                        PEER_INDEX_TABLE = m.data['peer_entries']
                        continue
                    elif subtype_code in [TD_V2_ST['RIB_IPV4_UNICAST'], TD_V2_ST['RIB_IPV6_UNICAST']]:
                        bgp_data['prefix'] = f"{m.data['prefix']}/{m.data['length']}"
                        for entry in m.data['rib_entries']:
                            peer = PEER_INDEX_TABLE[entry['peer_index']]
                            bgp_data['peer_ip'] = peer['peer_ip']
                            bgp_data['peer_as'] = peer['peer_as']
                            path_attrs = entry['path_attributes']
                            
                            # --- THIS IS THE FIX ---
                            # Pass the event_type 'B' to the format_row function
                            row = format_row(bgp_data, path_attrs, 'B')
                            
                            if row:
                                writer.writerow(row)
                                row_count += 1

                elif mrt_type_code == MRT_T['BGP4MP']:
                    subtype_code = list(m.data['subtype'])[0]
                    if subtype_code in [BGP4MP_ST['BGP4MP_MESSAGE'], BGP4MP_ST['BGP4MP_MESSAGE_AS4']]:
                        bgp_message = m.data['bgp_message']
                        bgp_data['peer_ip'] = m.data['peer_ip']
                        bgp_data['peer_as'] = m.data['peer_as']
                        path_attrs = bgp_message.get('path_attributes', [])

                        # Announcements (A)
                        if bgp_message.get('nlri'):
                            for nlri in bgp_message['nlri']:
                                bgp_data['prefix'] = f"{nlri['prefix']}/{nlri['length']}"
                                row = format_row(bgp_data, path_attrs, 'A')
                                if row:
                                    writer.writerow(row)
                                    row_count += 1
                        for attr in path_attrs:
                            if list(attr['type'])[0] == BGP_ATTR_T['MP_REACH_NLRI'] and 'nlri' in attr['value']:
                                for nlri in attr['value']['nlri']:
                                    bgp_data['prefix'] = f"{nlri['prefix']}/{nlri['length']}"
                                    row = format_row(bgp_data, path_attrs, 'A')
                                    if row:
                                        writer.writerow(row)
                                        row_count += 1
                        
                        # Withdrawals (W)
                        if bgp_message.get('withdrawn_routes'):
                            for withdrawn in bgp_message['withdrawn_routes']:
                                prefix = f"{withdrawn['prefix']}/{withdrawn['length']}"
                                writer.writerow([bgp_data['timestamp'], 'W', bgp_data['peer_ip'], bgp_data['peer_as'], prefix, '', '', ''])
                                row_count += 1
                        for attr in path_attrs:
                            if list(attr['type'])[0] == BGP_ATTR_T['MP_UNREACH_NLRI'] and 'withdrawn_routes' in attr['value']:
                                for withdrawn in attr['value']['withdrawn_routes']:
                                    prefix = f"{withdrawn['prefix']}/{withdrawn['length']}"
                                    writer.writerow([bgp_data['timestamp'], 'W', bgp_data['peer_ip'], bgp_data['peer_as'], prefix, '', '', ''])
                                    row_count += 1

            except (KeyError, IndexError, TypeError):
                # This will now only skip genuinely malformed records
                continue

            if record_count % 100000 == 0:
                print(f" ... processed {record_count} MRT records, wrote {row_count} rows")

    print(f"\nDone! Processed {record_count} total MRT records.")
    print(f"Extracted {row_count} data rows to '{output_file_path}'.")


def format_row(bgp_data, path_attributes, event_type):
    as_path_str = ''
    origin = ''
    next_hop = ''

    for attr in path_attributes:
        attr_type = list(attr['type'])[0]
        if attr_type == BGP_ATTR_T['AS_PATH']:
            path_list = []
            for seg in attr['value']:
                if list(seg['type'])[0] == AS_PATH_SEG_T['AS_SEQUENCE']:
                    path_list.extend(seg['value'])
            as_path_str = ' '.join(path_list)
        elif attr_type == BGP_ATTR_T['ORIGIN']:
            origin_code = list(attr['value'])[0]
            origin = ORIGIN_T[origin_code]
        elif attr_type == BGP_ATTR_T['NEXT_HOP']:
            next_hop = attr['value']
        elif attr_type == BGP_ATTR_T['MP_REACH_NLRI']:
            if 'next_hop' in attr['value'] and attr['value']['next_hop']:
                next_hop = attr['value']['next_hop'][0]

    if as_path_str:
        return [
            bgp_data['timestamp'], event_type, bgp_data['peer_ip'],
            bgp_data['peer_as'], bgp_data['prefix'], as_path_str,
            origin, next_hop
        ]
    return None

def main():
    parser = argparse.ArgumentParser(description='Extracts key BGP data from an MRT file to a CSV.')
    parser.add_argument('mrt_file', help='Path to the input MRT file')
    parser.add_argument('-o', '--output', default='data/bgp_data.csv', help='Path to the output CSV file')
    args = parser.parse_args()
    extract_bgp_data(args.mrt_file, args.output)

if __name__ == '__main__':
    main()