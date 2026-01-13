import pandas as pd
import numpy as np

def assert_df_equal(df_old, df_new, name="DF"):
    """
    Compares two dataframes and prints detailed differences.
    """
    try:
        pd.testing.assert_frame_equal(df_old, df_new, check_like=True)
        print(f"[PASS] {name} identical")
    except AssertionError as e:
        print(f"[FAIL] {name} mismatch:")
        print(e)
        print("OLD COLUMNS:", df_old.columns)
        print("NEW COLUMNS:", df_new.columns)
        print("OLD HEAD:\n", df_old.head())
        print("NEW HEAD:\n", df_new.head())
        raise e

def assert_list_equal(a, b, name):
    if a == b:
        print(f"[PASS] {name} identical")
    else:
        print(f"[FAIL] {name} mismatch")
        print("OLD:", a[:20])
        print("NEW:", b[:20])
        raise AssertionError(name)

def assert_array_equal(a, b, name):
    if np.array_equal(a, b):
        print(f"[PASS] {name} identical")
    else:
        print(f"[FAIL] {name} mismatch")
        diff = np.where(a != b)
        print("First 20 diffs:", diff[:20])
        raise AssertionError(name)

def assert_vocab_equal(old, new, name="Vocab"):
    # check equal size
    assert len(old.idx_to_word) == len(new.idx_to_word), (
        f"{name}: size mismatch "
        f"{len(old.idx_to_word)} != {len(new.idx_to_word)}"
    )
    
    # check index → token mapping identical
    assert old.idx_to_word == new.idx_to_word, (
        f"{name}: idx2word mismatch"
    )
    
    # check token → index mapping identical
    assert old.word_to_idx == new.word_to_idx, (
        f"{name}: word2idx mismatch"
    )

def assert_records_equal(old_records, new_records, name="Records"):
    assert len(old_records) == len(new_records), (
        f"{name}: patient count mismatch "
        f"{len(old_records)} != {len(new_records)}"
    )

    for i, (old_patient, new_patient) in enumerate(zip(old_records, new_records)):

        assert len(old_patient) == len(new_patient), (
            f"{name}: visit count mismatch for patient {i} "
            f"{len(old_patient)} != {len(new_patient)}"
        )

        for v, (old_visit, new_visit) in enumerate(zip(old_patient, new_patient)):
            assert len(old_visit) == len(new_visit) == 3, (
                f"{name}: visit {v} structure invalid for patient {i}"
            )

            # Compare diag / med / proc lists
            for j in range(3):
                assert old_visit[j] == new_visit[j], (
                    f"{name}: mismatch at patient {i}, visit {v}, field {j}\n"
                    f"old={old_visit[j]}\n"
                    f"new={new_visit[j]}"
                )