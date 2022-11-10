import java.io.IOException;
import java.io.File;

public class processBuilder {
    public static void main(String[] args) throws IOException, InterruptedException {

    ProcessBuilder pb = new ProcessBuilder("date");

    pb.directory(new File("/host"));
    File log = new File("/host/log");
    pb.redirectErrorStream(true);
    pb.redirectOutput(ProcessBuilder.Redirect.appendTo(log));

    Process process = pb.start();

    // String result = new String(process.getInputStream().readAllBytes());
    // System.out.printf("%s", result);

    int ret = process.waitFor();
    System.out.printf("Child process exited with code: %d\n", ret);
    }
}
