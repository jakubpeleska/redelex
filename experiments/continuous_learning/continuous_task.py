import pandas as pd

from relbench.base import EntityTask, Table

class ContinuousWrapper:
    def __init__(self, task: EntityTask):
        self.task = task

        train_table = self.task.get_table("train", mask_input_cols=False)
        val_table = self.task.get_table("val", mask_input_cols=False)
        test_table = self.task.get_table("test", mask_input_cols=False)

        df = (
            pd.concat([train_table.df, val_table.df, test_table.df], ignore_index=True)
            .sort_values(train_table.time_col)
            .reset_index(drop=True)
        )

        self.full_table = Table(
            df=df,
            fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
            pkey_col=train_table.pkey_col,
            time_col=train_table.time_col,
        )

    def get_table(self, start: pd.Timestamp, end: pd.Timestamp):
        mask = (self.full_table.df[self.full_table.time_col] >= start) & (
            self.full_table.df[self.full_table.time_col] < end
        )
        return Table(
            df=self.full_table.df[mask].reset_index(drop=True),
            fkey_col_to_pkey_table=self.full_table.fkey_col_to_pkey_table,
            pkey_col=self.full_table.pkey_col,
            time_col=self.full_table.time_col,
        )

    def get_splits(self, val_delta: pd.Timedelta = None):
        if val_delta is None:
            val_delta = (
                self.task.dataset.test_timestamp - self.task.dataset.val_timestamp
            )

        timestamps = self.full_table.df[self.full_table.time_col].unique()

        splits = [self.task.dataset.test_timestamp, self.task.dataset.val_timestamp]

        previous_timestamp = self.task.dataset.val_timestamp
        for timestamp in reversed(timestamps):
            if timestamp + val_delta <= previous_timestamp:

                splits.append(timestamp)
                previous_timestamp = timestamp

        splits.append(timestamps[0])
        splits.reverse()

        min_split_len = int(
            0.1
            * len(
                self.get_table(
                    start=self.task.dataset.val_timestamp,
                    end=self.task.dataset.test_timestamp,
                )
            )
        )
        filtered_splits = [splits[0]]
        for split in splits[1:]:
            split_len = len(self.get_table(start=filtered_splits[-1], end=split))
            if split_len >= min_split_len:
                filtered_splits.append(split)

        return filtered_splits