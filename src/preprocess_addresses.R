
# Load required libraries for Excel I/O
library(readxl)
library(writexl)


# Main function to preprocess Cuban-style addresses from Excel
preprocess_addresses <- function(input_file, output_file, auto_fill_second_cross = FALSE) {
    # Read the input Excel file into a data frame
    df <- read_excel(input_file)

    # Identify the address column (supports several possible names)
    address_cols <- c("address", "direccion", "direccion_particular",  "ubicacion", "localizacion")
    address_col <- address_cols[address_cols %in% names(df)][1]
    if (is.na(address_col)) {
        stop("No address column found. Expected one of: address, direccion, ubicacion, localizacion")
    }

    # Identify Consejo Popular/CP/Popular Council columns
    cp_col_variants <- c("consejo_popular", "cp", "pc", "popular_council", "consejo", "consejo popular", "popular council", "council", "pop council", "consejo_p", "cp_name", "cp_nombre", "nombre_cp", "nombre_consejo", "consejo_p_name")
    cp_col <- cp_col_variants[cp_col_variants %in% tolower(names(df))][1]
    # If not found, try partial match
    if (is.na(cp_col)) {
        cp_col <- names(df)[grepl("consejo|cp|popular|council", tolower(names(df)))[1]]
    }


    # Normalize and clean a raw address string
    normalize_text <- function(x) {
        # Lowercase, trim, remove accents, replace punctuation with spaces, remove linebreaks, collapse spaces
        if (is.na(x) || x == "") return(NA_character_)
        x <- tolower(trimws(x))
        x <- iconv(x, from = "UTF-8", to = "ASCII//TRANSLIT")
        x <- gsub("[^a-z0-9#/ -]", " ", x) # keep only allowed chars
        x <- gsub("\\n|\\r", " ", x)
        x <- gsub("\\s+", " ", x)
        trimws(x)
    }


    # Normalize a street token (e.g., 'calle 23' -> 'c-23', 'avenida 54' -> 'a-54')
    normalize_street_token <- function(token) {
        # Remove spaces
        t <- gsub("\\s+", "", token)
        # Normalize all variants of 'calle' to 'c'
        t <- gsub("^calle", "c", t)
        t <- gsub("^call", "c", t)
        t <- gsub("^cal", "c", t)
        t <- gsub("^ca", "c", t)
        # Normalize all variants of 'avenida' to 'a' (longest first)
        t <- gsub("^avenida", "a", t)
        t <- gsub("^avenid", "a", t)
        t <- gsub("^aveni", "a", t)
        t <- gsub("^aven", "a", t)
        t <- gsub("^ave", "a", t)
        t <- gsub("^av", "a", t)
        # Replace '/' with '-'
        t <- gsub("/", "-", t)
        # If token is just a prefix, return NA
        if (t %in% c("a", "c", "e")) return(NA_character_)
        # Ensure prefix is followed by a dash
        if (!grepl("^[ace]-", t)) {
            t <- gsub("^([ace])(?=[0-9a-z])", "\\1-", t, perl = TRUE)
        }
        # Clean up repeated dashes
        t <- gsub("^([ace])-+", "\\1-", t)
        t <- gsub("-{2,}", "-", t)
        t
    }


    # Output vectors for parsed address components
    main_street_vec <- rep(NA_character_, nrow(df))
    cross_streets_vec <- rep(NA_character_, nrow(df))
    house_number_vec <- rep(NA_character_, nrow(df))


    # Main parsing loop: process each address row
    for (i in seq_len(nrow(df))) {
        address <- normalize_text(df[[address_col]][i])

        # Remove Consejo Popular/CP/Popular Council value from address if present
        if (!is.na(cp_col) && cp_col %in% tolower(names(df))) {
            cp_val <- normalize_text(df[[which(tolower(names(df)) == cp_col)]][i])
            if (!is.na(cp_val) && nzchar(cp_val) && grepl(cp_val, address, ignore.case = TRUE)) {
                address <- gsub(cp_val, "", address, ignore.case = TRUE)
                address <- trimws(address)
            }
        }

        # Remove everything after 'area' (if present)
        if (!is.na(address) && grepl("area", address, ignore.case = TRUE)) {
            address <- sub("area.*$", "", address, ignore.case = TRUE)
            address <- trimws(address)
        }
        # Remove all occurrences of 'cienfuegos' (if present)
        if (!is.na(address) && grepl("cienfuegos", address, ignore.case = TRUE)) {
            address <- gsub("cienfuegos", "", address, ignore.case = TRUE)
            address <- trimws(address)
        }


        # Initialize output variables for this row
        main_street <- NA
        cross_streets <- NA
        house_number <- NA


        # Skip processing if address contains or starts with 'EDIF', 'EDF', 'APT', or 'EDIFICIO' (case-insensitive)
        # This is to avoid parsing building/apt-only addresses
        if (!is.na(address) && grepl("^(edificio|edif|edf|apt)|\\bedif\\b|\\bedf\\b|\\bapt\\b", address, ignore.case = TRUE)) {
            main_street_vec[i] <- NA
            cross_streets_vec[i] <- NA
            house_number_vec[i] <- NA
            next
        }


        if (!is.na(address)) {

            # If address contains only letters and spaces (no digits), treat as a named street
            if (grepl("^[a-z ]+$", address) && !grepl("[0-9]", address)) {
                main_street <- trimws(address)
                main_street_vec[i] <- main_street
                cross_streets_vec[i] <- NA
                house_number_vec[i] <- NA
                next
            }


            # Extract main street(s) using regex for Cuban street prefixes
            # Matches: 'calle 23', 'avenida libertad', 'ave 32', etc.
            street_pattern <- "(?:calle|c|avenida|avenid|aveni|aven|ave|av|a)\\s*[-/ ]?\\s*([a-z0-9]+)"
            matches <- gregexpr(street_pattern, address, perl = TRUE)
            match_pos <- matches[[1]]
            match_len <- attr(matches[[1]], "match.length")
            streets <- character()
            if (match_pos[1] > 0) {
                for (j in seq_along(match_pos)) {
                    if (match_pos[j] > 0) {
                        matched_str <- substr(address, match_pos[j], match_pos[j] + match_len[j] - 1)
                        streets <- c(streets, normalize_street_token(matched_str))
                    }
                }
            }


            # Extract cross streets (e.g., '2da y 3ra', '2 y 3', etc.)
            cross_pat <- "([0-9]{1,2})(?:da|ra|ta|na)?\\s*[yY]\\s*([0-9]{1,2})(?:da|ra|ta|na)?"
            cross_match <- regexpr(cross_pat, address, perl = TRUE)
            cross_str <- NA_character_
            if (cross_match[1] > 0) {
                cross_groups <- regmatches(address, regexec(cross_pat, address, perl = TRUE))[[1]]
                if (length(cross_groups) >= 3) {
                    cross_str <- paste0(cross_groups[2], " y ", cross_groups[3])
                }
            }


            # Look for 4+ digit house number (with possible trailing letter), e.g., '#5903', 'no 5903', '5903a'
            house_pat <- "(?:#|\\bno\\.? ?|\\bno)([0-9]{4,}[a-zA-Z]?)"
            house_match <- regexpr(house_pat, address, perl = TRUE)
            house_number_str <- NA_character_
            if (house_match[1] > 0) {
                house_number_str <- regmatches(address, house_match)
                if (is.list(house_number_str)) house_number_str <- house_number_str[[1]][2]
            } else {
                house_pat2 <- "([0-9]{4,}[a-zA-Z]?)"
                house_match2 <- regexpr(house_pat2, address, perl = TRUE)
                if (house_match2[1] > 0) {
                    house_number_str <- regmatches(address, house_match2)
                }
            }


            # If main street and 4+ digit house number found (and no explicit cross_street), auto-fill cross_streets and house_number
            if (length(streets) > 0 && !is.na(house_number_str) && nchar(gsub("[^0-9]", "", house_number_str)) >= 4) {
                main_street <- streets[1]
                # Use first two digits for cross_streets, last two for house_number
                hn_digits <- gsub("[^0-9]", "", house_number_str)
                first_cross <- as.numeric(substr(hn_digits, 1, 2))
                second_cross <- sprintf("%02d", first_cross + 2)
                cross_streets <- paste0(sprintf("%02d", first_cross), " y ", second_cross)
                hn_len <- nchar(hn_digits)
                house_number <- substr(hn_digits, hn_len-1, hn_len)
                # Append trailing letter if present
                if (grepl("[a-zA-Z]$", house_number_str)) {
                    house_number <- paste0(house_number, substr(house_number_str, nchar(house_number_str), nchar(house_number_str)))
                }
                main_street_vec[i] <- main_street
                cross_streets_vec[i] <- cross_streets
                house_number_vec[i] <- house_number
                next
            }


            # Extract house number (e.g., #1234 or no 1234, with optional trailing letter)
            house_number_str <- NA_character_
            house_match <- regexpr("(?:#|\\bno\\.?)[ ]*([0-9]+[a-z]?)", address, perl = TRUE)
            if (house_match[1] > 0) {
                house_number_str <- gsub("[^0-9]", "", trimws(regmatches(address, house_match)))
            } else {
                # Fallback: match any standalone number (3+ digits)
                house_match2 <- regexpr("([0-9]{3,})", address, perl = TRUE)
                if (house_match2[1] > 0) {
                    house_number_str <- regmatches(address, house_match2)
                }
            }


            # If we have a street, try to extract house number and cross_streets from it
            if (length(streets) > 0) {
                main_street <- streets[1]
                main_street_num <- NA_character_
                main_street_num_match <- regexpr("[0-9]+", main_street)
                if (!is.na(main_street_num_match[1]) && main_street_num_match[1] > 0) {
                    main_street_num <- regmatches(main_street, main_street_num_match)
                }
                # If house_number_str is a 4+ digit number, use for cross_streets/house_number
                if (!is.na(house_number_str) && nzchar(house_number_str) && nchar(house_number_str) >= 4) {
                    cross_streets <- substr(house_number_str, 1, 2)
                    house_number <- substr(house_number_str, nchar(house_number_str)-1, nchar(house_number_str))
                } else if (!is.na(main_street_num) && nzchar(main_street_num) && nchar(main_street_num) >= 4) {
                    cross_streets <- substr(main_street_num, 1, 2)
                    house_number <- substr(main_street_num, nchar(main_street_num)-1, nchar(main_street_num))
                } else if (!is.na(house_number_str) && nzchar(house_number_str)) {
                    house_number <- house_number_str
                }
            }


            # If cross_streets not set, try to extract from multiple street tokens or fallback patterns
            if (is.na(cross_streets)) {
                if (length(streets) >= 3) {
                    cross_streets <- paste(streets[2], "y", streets[3])
                } else if (length(streets) == 2) {
                    cross_streets <- streets[2]
                } else if (!is.na(main_street)) {
                    # Try to extract cross streets from fallback patterns
                    cross_match_three <- regexpr("/\\s*([0-9]+)\\s*/\\s*([0-9]+)\\s*y\\s*([0-9]+)", address, perl = TRUE)
                    if (cross_match_three[1] > 0) {
                        cross_str <- regmatches(address, cross_match_three)
                        nums <- unlist(regmatches(cross_str, gregexpr("[0-9]+", cross_str)))
                        if (length(nums) >= 3) {
                            cross_streets <- paste(nums[2], "y", nums[3])
                        }
                    } else {
                        cross_match_two <- regexpr("/\\s*([0-9]+)\\s*y\\s*([0-9]+)", address, perl = TRUE)
                        if (cross_match_two[1] > 0) {
                            cross_str <- regmatches(address, cross_match_two)
                            nums <- unlist(regmatches(cross_str, gregexpr("[0-9]+", cross_str)))
                            if (length(nums) >= 2) {
                                cross_streets <- paste(nums[1], "y", nums[2])
                            }
                        }
                    }
                }
            }


            # Fallback: try to extract a named street (e.g., 'castillo 105 caonao')
            if (is.na(main_street)) {
                name_match <- regexpr("[a-z]+\\s*[0-9]+", address, perl = TRUE)
                if (name_match[1] > 0) {
                    main_street <- regmatches(address, name_match)
                }
            }
        }

        # Optionally auto-fill second cross street if only one is present (e.g., '41' -> '41 y 43')
        if (auto_fill_second_cross && !is.na(cross_streets) && grepl("^\\d{2}$", cross_streets)) {
            first_cross <- as.numeric(cross_streets)
            if (!is.na(first_cross)) {
                second_cross <- first_cross + 2
                cross_streets <- paste0(sprintf("%02d", first_cross), " y ", sprintf("%02d", second_cross))
            }
        }
        # Store results for this row
        main_street_vec[i] <- main_street
        cross_streets_vec[i] <- cross_streets
        house_number_vec[i] <- house_number
    }


    # Add new columns for parsed address components next to the original address column
    col_index <- match(address_col, names(df))
    new_cols <- data.frame(CALLE_O_AVENIDA = main_street_vec,
                           ENTRE_CALLE_O_AVENIDAS = cross_streets_vec,
                           NUMERO_DE_VIVIENDA = house_number_vec,
                           stringsAsFactors = FALSE)
    if (col_index == ncol(df)) {
        output_df <- cbind(df, new_cols)
    } else {
        output_df <- cbind(df[1:col_index], new_cols, df[(col_index + 1):ncol(df)])
    }

    # Save the output data frame to an Excel file
    write_xlsx(output_df, output_file)
}


# Example usage:
# preprocess_addresses('input_addresses.xlsx', 'output_addresses.xlsx')
