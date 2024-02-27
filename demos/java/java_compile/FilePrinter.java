import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class FilePrinter {

    public static void main(String[] args) {
        String filePath = "file.txt";

        File file = new File(filePath);
        
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
                System.err.println("Error closing reader: " + e.getMessage());
            }
        }
    }
}
