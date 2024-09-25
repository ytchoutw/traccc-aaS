import argparse
import sys

import numpy as np
import pandas as pd
import tritonclient.http as httpclient

def main():
    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count, ssl=False
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    print("\n=========")
    async_requests = []

    # input0_data = pd.read_csv(FLAGS.filename).to_numpy(dtype=np.float64)
    # make some dummy data
    input0_data = np.zeros((6, 6), dtype=np.float64)
    print("Sending request to batching model: input = {}".format(input0_data))
    inputs = [httpclient.InferInput("FEATURES", input0_data.shape, "FP64")]
    inputs[0].set_data_from_numpy(input0_data)
    async_requests.append(triton_client.async_infer(f"traccc-gpu", inputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference
        # request. This call will block till the server responds.
        result = async_request.get_result()
        print("Response: {}".format(result.get_response()))
        print("OUTPUT = {}".format(result.as_numpy("LABELS")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default="event000000000-cells.csv",
        help="Input file name. Default is event000000000-cells.csv.",
    )
    FLAGS = parser.parse_args()

    main()