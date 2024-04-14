package daislab.cspg;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.nio.file.Files;

/*
 * Class that creates and writes a CSV file
*/
public class CsvWriter {
    private CSVPrinter csvPrinter;

    public CsvWriter(final String directory, final String filename, String[] header) {
        final CSVFormat csvFormat = CSVFormat.DEFAULT.builder().setHeader(header).build();
        final String filepath = Paths.get(directory, filename).toString();
        System.out.println("egrapsa ton gamwheader" + header);

        // Create log directory if not exist
        try {
            Files.createDirectories(Paths.get(directory));
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            csvPrinter = new CSVPrinter(new FileWriter(filepath), csvFormat);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
    }

    // Write a single row to the CSV file and flush
    public void writeRow(Object[] data) {
        try {
            csvPrinter.printRecord(data);
            csvPrinter.flush();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
    }

    public void close() {
        try {
            csvPrinter.close();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
    }
}
