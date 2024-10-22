"""
First we need to run `bash get_all_error_predict.sh` to save all the error prediction results to a text file.

"""
import os
import fire
import matplotlib.pyplot as plt


def get_error_predict_list(file_path: str) -> list:
    with open(file_path, "r") as f:
        lines = f.readlines()
    error_predict_list = []
    for line in lines:
        error_predict_list.append(line.strip())
    return error_predict_list


def get_error_type_from_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "llc: error:" not in line:
                continue
            elif line.count("error:") == 1:
                if line.find("input module cannot be verified")>-1:
                    return "input module cannot be verified"
            elif line.count("error:") == 2:
                error_type = line.split("error:")[-1].strip()
                return error_type
            else:
                print(f"error: {line}")
    return None


def get_error_type(error_predict_list: list) -> dict:
    error_type_dict = {}
    for error_predict in error_predict_list:
        error_type = get_error_type_from_file(error_predict)
        if error_type is None:
            continue
        if error_type in error_type_dict:
            error_type_dict[error_type] += 1
        else:
            error_type_dict[error_type] = 1
    return error_type_dict


def analyze_error_type(file_path):
    error_predict_list = get_error_predict_list(file_path)
    error_type_dict = get_error_type(error_predict_list)
    new_error_dict = post_processing_error_dict(error_type_dict)
    draw_error_type_pie(new_error_dict)


def post_processing_error_dict(error_dict_count: dict) -> dict:
    error_type_list = [
        "input module cannot be verified",
        "expected top-level entity",
        "use of undefined value",
        "label expected to be numbered",
        "instruction forward referenced with type 'label'",
        "alignment is not a power of two",
        "instruction expected to be numbered",
        "label expected to be numbered",
        "defined with type",
        "invalid getelementptr indices",
        "value doesn't match function result type",
        "instruction forward referenced with type",
        "redefinition of function",
        "expected value token",
        "is not a basic block",
        "multiple definition of local value named",
        "unable to create block numbered",
        "base element of getelementptr must be sized",
        "use of undefined comdat",
        "invalid shufflevector operands",
        "invalid use of function-local name",
        "expected instruction opcode"
    ]
    new_error_dict = {}
    for k, v in error_dict_count.items():
        for error_type in error_type_list:
            if error_type in k:
                if error_type in new_error_dict:
                    new_error_dict[error_type] += v
                else:
                    new_error_dict[error_type] = v
                break
    return new_error_dict



def draw_error_type_pie(error_dict_count: dict):
    error_count_list = [(k, v) for k,v in error_dict_count.items()]
    sum = 0
    for k, v in error_count_list:
        sum += v
    error_count_list.sort(key=lambda x: x[1], reverse=False)
    other, idx = 0, 0
    for i, (k, v) in enumerate(error_count_list):
        if v / sum < 0.05:
            other += v
        else:
            idx = i
            break
    error_count_list = error_count_list[idx:]
    error_count_list.append(("other", other))
    labels = [k for k, v in error_count_list]
    sizes = [v for k, v in error_count_list]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of Error Types")
    plt.savefig(os.path.join(os.path.dirname(__file__), "error_type_pie.png"))
    



if __name__ == "__main__":
    fire.Fire(analyze_error_type)