library(readxl)
library(writexl)

preprocess_addresses <- function(input_file, output_file) {
    # Read the input Excel file
    df <- read_excel(input_file)

    # Initialize a new data frame for the output
    output_df <- data.frame(CALLE_O_AVENIDA = character(),
                             ENTRE_CALLE_O_AVENIDAS = character(),
                             NUMERO_DE_VIVIENDA = numeric(),
                             stringsAsFactors = FALSE)

    # Process each row in the input data frame
    for (i in 1:nrow(df)) {
        address <- tolower(trimws(gsub("[[:space:]]+|[áéíóúÁÉÍÓÚ]", "", df[i, c("address", "direccion", "ubicacion", "localizacion")]))[1])  # Handle multiple possible column names

        # Initialize variables
        main_street <- NA
        cross_streets <- NA
        house_number <- NA

        # Extract components from the address using intelligent pattern matching
        # Format: main_street main_street_number cross_street cross_street_number house_number
        # Street names can be: av, ave, avenida, c, calle, etc.
        
        # Pattern to match street abbreviations/names followed by numbers
        street_pattern <- "(?:av|ave|avenida|c|calle|ca|cal)[a-z]*([0-9]+)"
        
        # Find all matches of street pattern in the address
        matches <- gregexpr(street_pattern, address, perl = TRUE)
        match_pos <- matches[[1]]
        match_len <- attr(matches[[1]], "match.length")
        
        if (match_pos[1] > 0) {
            # Extract all matched street components
            streets <- character()
            street_numbers <- numeric()
            
            for (j in seq_along(match_pos)) {
                if (match_pos[j] > 0) {
                    matched_str <- substr(address, match_pos[j], match_pos[j] + match_len[j] - 1)
                    streets[j] <- matched_str
                    # Extract the number from the matched string
                    street_num <- as.numeric(gsub("[^0-9]", "", matched_str))
                    street_numbers[j] <- street_num
                }
            }
            
            # First street is the main street
            if (length(streets) > 0) {
                main_street <- streets[1]
            }
            
            # Second and third streets are the cross streets
            if (length(streets) > 1) {
                cross_streets <- paste(streets[2], "y", streets[3], sep = " ")
            }
            
            # Extract the house number (last sequence of digits after the street patterns)
            remaining <- substr(address, max(match_pos + match_len), nchar(address))
            house_number_str <- gsub("[^0-9]", "", remaining)
            if (house_number_str != "") {
                house_number <- as.numeric(house_number_str)
            }
        }

        # Append the processed row to the output data frame
        output_df <- rbind(output_df, data.frame(CALLE_O_AVENIDA = main_street,
                                                   ENTRE_CALLE_O_AVENIDAS = cross_streets,
                                                   NUMERO_DE_VIVIENDA = house_number,
                                                   stringsAsFactors = FALSE))
    }

    # Save the output data frame to an Excel file
    write_xlsx(output_df, output_file)
}

# Example usage:
# preprocess_addresses('input_addresses.xlsx', 'output_addresses.xlsx')
