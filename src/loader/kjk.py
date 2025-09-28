def load_analysis(random_state: int,
                  eval_size: float,
                  split: bool,
                  old_split_config_date: str | None
                  ) -> pd.DataFrame:
    """
    Run the analysis pipeline, save results and metadata, and optionally split the dataset.
    """
    if split is True and eval_size is None:
        raise ValueError("eval_size must be set when split is True.")

    if old_split_config_date is not None and split is True:
        raise ValueError("Cannot use old_split_config_date when split is True.")

    # 1) Generate a timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = create_run_folder(timestamp)
    logging.info("Created run directory %s", run_dir)

    # 2) Execute analysis
    analysis = run_analysis_with_configuration_parameters(run_dir)
    metrics_df = analysis.get_metrics_dataframe()
    metrics_df = add_metadata_to_metrics(metrics_df)
    valid_metrics_df = filter_invalid_subjects(metrics_df)

    # 3) Split the dataset if required
    train_subject_ids, eval_subject_ids = get_split_ids(eval_size, old_split_config_date, random_state, split,
                                                        valid_metrics_df)

    # 4) Save results and metadata
    save_path = save_results(valid_metrics_df, run_dir, timestamp, random_state, eval_size, train_subject_ids, eval_subject_ids,
                 split,
                 old_split_config_date)

    return valid_metrics_df, save_path