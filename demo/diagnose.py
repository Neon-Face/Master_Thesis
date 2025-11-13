import sys
from mrtparse import Reader, MRT_T, TD_V2_ST

def diagnose_dump_file(mrt_file_path):
    """
    Reads the first few records of an MRT dump file to diagnose its structure.
    """
    print(f"--- Starting Diagnostics on DUMP file: {mrt_file_path} ---\n")
    
    record_limit = 50
    print(f"Inspecting the first {record_limit} records...\n")

    found_peer_index = False
    found_rib_entry = False

    try:
        for i, m in enumerate(Reader(mrt_file_path)):
            if i >= record_limit:
                break
            
            if m.err:
                print(f"Skipping record #{i+1} due to mrtparse error: {m.err_msg}")
                continue

            record = m.data
            
            if 'type' not in record:
                print(f"Record #{i+1}: SKIPPED (no 'type' field).")
                continue

            mrt_type_code = list(record['type'])[0]
            mrt_type_name = record['type'][mrt_type_code]
            
            print(f"--- Record #{i+1}: Found MRT Type: {mrt_type_name} ---")

            # Check for the old TABLE_DUMP V1 format
            if mrt_type_code == MRT_T['TABLE_DUMP']:
                print("  [INFO] This is a TABLE_DUMP (V1) format file.")
                print("         The main script is designed for V2 and needs modification for V1.")
                # Print a snippet of the V1 structure
                print(f"  > Peer: {record.get('peer_as')}, Prefix: {record.get('prefix')}/{record.get('length')}")
                found_rib_entry = True

            # Check for the new TABLE_DUMP V2 format
            elif mrt_type_code == MRT_T['TABLE_DUMP_V2']:
                print("  [INFO] This is a TABLE_DUMP_V2 (V2) format file.")
                subtype_code = list(record['subtype'])[0]
                subtype_name = record['subtype'][subtype_code]
                print(f"  > Subtype: {subtype_name}")

                if subtype_code == TD_V2_ST['PEER_INDEX_TABLE']:
                    print("  [SUCCESS] Found the PEER_INDEX_TABLE. This is the 'table of contents'.")
                    found_peer_index = True
                
                elif subtype_code in [TD_V2_ST['RIB_IPV4_UNICAST'], TD_V2_ST['RIB_IPV6_UNICAST']]:
                    print(f"  > Contains {len(record.get('rib_entries', []))} route entries for prefix {record.get('prefix')}/{record.get('length')}")
                    found_rib_entry = True

            print("-" * 40)

            # Stop after we have some info
            if found_rib_entry:
                break
        
        print("\n--- Diagnostics Complete ---")
        if not found_peer_index and found_rib_entry and mrt_type_code == MRT_T['TABLE_DUMP_V2']:
             print("\n[CRITICAL] Found RIB entries but DID NOT find a PEER_INDEX_TABLE first.")
             print("           This is the reason the main script is failing.")
        elif not found_rib_entry:
             print("\n[WARNING] Could not find any recognizable RIB entries in the first 50 records.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_dump.py <path_to_mrt_file>")
        sys.exit(1)
    
    mrt_file = sys.argv[1]
    diagnose_dump_file(mrt_file)
