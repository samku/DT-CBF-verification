def generate_file_path(step_number,base_filename, current_directory):
    folder_name = f"{base_filename}"  
    filename = f"{step_number}.pkl" 
    folder_path = current_directory / "CBF_results" / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)  
    file_path = folder_path / filename  
    return file_path