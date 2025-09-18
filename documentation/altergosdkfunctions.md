def extract_altergo_parameters() -> dict:
    """
    Extract the Altergo parameters from the local arguments or Altergo system parameters
    """

    altergoArguments = None
    pathToSettings = os.path.join(os.path.dirname(sys.argv[0]), "altergo-settings.json")

    # check if sys.argv size is 2 or above. If no, raise exception with the specific error saying "could not read arguments"
    if len(sys.argv) < 2:
        raise Exception("could not read arguments")
    
    argumentsParameters = sys.argv[1]

    if argumentsParameters != "dev-parameters.json":
        altergoArguments = None
        try:
            # Try to interpret the input as JSON string (Altergo platform)
            altergoArguments = json.loads(argumentsParameters)
            print("Arguments interpreted as inline JSON (Altergo platform).")
        except json.JSONDecodeError:
            def load_arguments_from_file(file_path: str) -> dict:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    raise Exception("Could not read arguments from file path provided.")
            # Fallback: likely running locally with file path
            print("This run is not from Altergo platform, but from local environment. Checking a file path to read the arguments...")
            altergoArguments = load_arguments_from_file(argumentsParameters)
            altergoArguments = json.loads(altergoArguments)

        try: 
            sent_configuration = altergoArguments["configurationValues"]
            expected_configuration = extract_default_altergo_settings_parameters(pathToSettings)

            def merge_configurations(sent_config, expected_config, parent_key=''):
                for key, value in expected_config.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    if key not in sent_config:
                        print(f"Warning: Missing key '{full_key}' in sent_configuration. Updating with default value: {value}")
                        sent_config[key] = value
                    elif isinstance(value, dict) and isinstance(sent_config[key], dict):
                        merge_configurations(sent_config[key], value, full_key)
                    else:
                        sent_config[key] = sent_config[key]
            merge_configurations(sent_configuration, expected_configuration)

            altergoArguments["configurationValues"] = sent_configuration
            return altergoArguments
        except Exception as e:
            raise Exception(f"Could not read configuration values from the Altergo settings file. Is it a valid JSON file? Error: {str(e)}")
        
    else:
        print("This run is not from Altergo platform, but from local environment. Checking a file path to read the arguments...")
        filePath = argumentsParameters
        try:
            with open(filePath) as f:
                altergoArguments = json.load(f)
        except Exception as e:
            # raise exception with the specific error saying "could not read arguments"
            raise Exception("Could not read arguments")
            
        # check that the altergoArguments contains the mandatory keys: altergoUserApiKey, altergoFactoryApi, altergoIotApi
        if 'altergoUserApiKey' not in altergoArguments:
            raise Exception("altergoUserApiKey is missing")
        
        if 'altergoFactoryApi' not in altergoArguments:
            raise Exception("altergoFactoryApi is missing")
        
        if 'altergoIotApi' not in altergoArguments:
            raise Exception("altergoIotApi is missing")
        
        altergoArguments['altergoProgramTaskId'] = 'dev'
        
        # check if at the path of the current executed file, there is a file named "dev-parameters.json"
        configurationValues = extract_default_altergo_settings_parameters(pathToSettings)
        altergoArguments['configurationValues'] = configurationValues
            
        return altergoArguments

def extract_default_altergo_settings_parameters(pathToSettings) -> dict:
    if os.path.exists(pathToSettings):
        configurationValues = {}
        
        def parse_configuration(template):
            result = {}
            for key, value in template.items():
                if isinstance(value, dict):
                    if 'valueDev' in value or 'default' in value:
                        # Prefer valueDev if provided; fallback to default
                        result[key] = value.get('valueDev', value.get('default'))
                    else:
                        # Recursively parse the nested dictionary
                        result[key] = parse_configuration(value)
                else:
                    # Handle non-dictionary values if necessary
                    result[key] = value
            return result

        try:
            with open(pathToSettings) as f:
                settings = json.load(f)
                if not "parameters" in settings:
                    raise Exception("Parameters key is missing in the settings file (.parameters)")
                
                configurationValues = parse_configuration(settings["parameters"])
                return configurationValues
        except Exception as e:
            # raise exception with the specific error saying "could not read arguments"
            raise Exception("Could not parse altergo-settings.json. Is it a valid JSON file?")
    else:
        raise Exception("Could not read altergo-settings.json. Does the file exist?")