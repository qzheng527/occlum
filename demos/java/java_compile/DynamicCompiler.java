import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

public class DynamicCompiler {

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();

        System.out.println("t1:"+ (System.currentTimeMillis()-start));
        int result = compiler.run(null, null, null, "FilePrinter.java");
        System.out.println("t2:"+ (System.currentTimeMillis()-start));

        if (result == 0) {
            System.out.println("Compilation successful.");
            runClass("./", "FilePrinter");
        } else {
            System.out.println("Compilation failed.");
        }

        System.out.println("t3:"+ (System.currentTimeMillis()-start));
    }

    public static void runClass(String classesDir, String className) {
        try {
            URL[] classURLs = { new File(classesDir).toURI().toURL() };
            try (URLClassLoader classLoader = new URLClassLoader(classURLs)) {
                Class<?> cls = classLoader.loadClass(className);
                Method main = cls.getMethod("main", String[].class);
                String[] mainArgs = new String[] {};
                main.invoke(null, (Object) mainArgs);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}