import javax.tools.*;
import java.io.*;
import java.lang.reflect.Method;
import java.net.*;
import java.util.*;
import java.security.SecureClassLoader;

public class InMemoryJavaCompilerExample {
    public static void main(String[] args) throws Exception {
        long start = System.currentTimeMillis();
        String sourceCode = "public class HelloWorld {" +
                            "  public static void main(String[] args) {" +
                            "    System.out.println(\"Hello, World!\");" +
                            "  }" +
                            "}";
        
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        JavaFileManager fileManager = new ForwardingJavaFileManager<JavaFileManager>(
                compiler.getStandardFileManager(diagnostics, null, null)) {
            private final Map<String, ByteArrayOutputStream> byteCodeMap = new HashMap<>();

            @Override
            public JavaFileObject getJavaFileForOutput(Location location, String className, JavaFileObject.Kind kind, FileObject sibling) throws IOException {
                if (kind == JavaFileObject.Kind.CLASS) {
                    return new SimpleJavaFileObject(URI.create("string:///" + className.replace('.', '/') + kind.extension), kind) {
                        @Override
                        public OutputStream openOutputStream() {
                            ByteArrayOutputStream baos = new ByteArrayOutputStream();
                            byteCodeMap.put(className, baos);
                            return baos;
                        }
                    };
                } else {
                    return super.getJavaFileForOutput(location, className, kind, sibling);
                }
            }

            @Override
            public ClassLoader getClassLoader(Location location) {
                return new SecureClassLoader() {
                    @Override
                    protected Class<?> findClass(String name) throws ClassNotFoundException {
                        ByteArrayOutputStream baos = byteCodeMap.get(name);
                        if (baos == null) {
                            throw new ClassNotFoundException(name);
                        }
                        byte[] byteCode = baos.toByteArray();
                        return super.defineClass(name, byteCode, 0, byteCode.length);
                    }
                };
            }
        };
        
        JavaCompiler.CompilationTask task = compiler.getTask(null, fileManager, diagnostics, null, null, Collections.singletonList(new SimpleJavaFileObject(URI.create("string:///HelloWorld.java"), JavaFileObject.Kind.SOURCE) {
            @Override
            public CharSequence getCharContent(boolean ignoreEncodingErrors) {
                return sourceCode;
            }
        }));

        System.out.println("t1:"+ (System.currentTimeMillis()-start));
        
        if (task.call()) {
            System.out.println("t2:"+ (System.currentTimeMillis()-start));
            ClassLoader classLoader = fileManager.getClassLoader(StandardLocation.CLASS_OUTPUT);
            Class<?> helloWorldClass = Class.forName("HelloWorld", true, classLoader);
            Method mainMethod = helloWorldClass.getDeclaredMethod("main", String[].class);
            mainMethod.invoke(null, (Object) new String[]{});
        } else {
            for (Diagnostic<?> diagnostic : diagnostics.getDiagnostics()) {
                System.out.format("Error on line %d in %s%n", diagnostic.getLineNumber(), diagnostic.getSource());
                System.out.println(diagnostic.getMessage(null));
            }
        }
        
        System.out.println("t3:"+ (System.currentTimeMillis()-start));
        fileManager.close();
    }
}
