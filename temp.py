import numpy as np

def create_bins_of_model_ids(df_len, nbr_clients=1):
    """
    Simulates records coming from different clients/models.  There is 1 model per client.
    This function creates a list of nbr_clients bins, where each bin is as close as equally sized
    as possible, such the list's length == df_len.

    :param df_len: (int) number of rows in the test dataset
    :param nbr_clients: (int) the number of distinct model IDs to generate (there is
        1 model per client), a.k.a. the number of bins

    :return: list of model IDs
    """
    bin_sizes = np.arange(df_len + nbr_clients - 1, df_len - 1, -1) // nbr_clients
    return [i for j in [bin_sizes[b]*[b] for b in range(len(bin_sizes))] for i in j]

print(create_bins_of_model_ids(103, 3))
