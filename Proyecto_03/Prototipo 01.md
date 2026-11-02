# 02/11 - Primer prototipo
Comenzamos como base los conceptos con nuestro tema: **Deporte junto a comunidad**, inspirado en los proyectos de ... 

Luego de una conversacion, decidimos hacer una instalacion donde cada persona colocara una cinta del color de su deporte y cuando estas cintas sean colocadas en la instaklacion suenen, mientras mas gente coloque sus cintas
estas emitiran distintos sonidos, las personas son conductivas en este trabajo al igual que los materiales:

- Nuestro primer prototipo esta hecho con una ESP32 junto a una bocina y cables, queremos añadir mas cosas si es necesario segun el proceso.

## Conexiones

| Elemento | Pin ESP32 |
|---|---|
| Cable 1 | GPIO 32 |
| Cable 2 | GPIO 33 |
| Parlante / buzzer | GPIO 25 |

El parlante va con su transistor y resistencias

## Código

```cpp
const int PIN_TOUCH1 = 32;
const int PIN_TOUCH2 = 33;
const int PIN_BUZZER = 25;

// Notas pentatónicas: siempre combinan
const int NOTA1 = 523;   // Do
const int NOTA2 = 659;   // Mi

int umbral1, umbral2;
bool tocando1 = false, tocando2 = false;

int calibrar(int pin) {
  long suma = 0;
  for (int i = 0; i < 50; i++) {
    suma += touchRead(pin);
    delay(10);
  }
  int reposo = suma / 50;
  Serial.printf("Pin %d -> reposo: %d  umbral: %d\n", pin, reposo, (int)(reposo * 0.7));
  return reposo * 0.7;
}

// Nota que se apaga suave, como una campanita
void tocarCampana(int freq, int ms) {
  ledcChangeFrequency(PIN_BUZZER, freq, 8);
  int pasos = ms / 10;
  for (int i = 0; i < pasos; i++) {
    float t = (float)i / pasos;
    int duty = 128 * (1 - t) * (1 - t);
    ledcWrite(PIN_BUZZER, duty);
    delay(10);
  }
  ledcWrite(PIN_BUZZER, 0);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  ledcAttach(PIN_BUZZER, 2000, 8);

  Serial.println("Calibrando... no toques los cables");
  umbral1 = calibrar(PIN_TOUCH1);
  umbral2 = calibrar(PIN_TOUCH2);
  Serial.println("Listo!");
}

void loop() {
  int v1 = touchRead(PIN_TOUCH1);
  int v2 = touchRead(PIN_TOUCH2);
  Serial.printf("C1: %d  C2: %d\n", v1, v2);

  bool t1 = v1 < umbral1;
  bool t2 = v2 < umbral2;

  // Suena solo en el instante en que empiezas a tocar
  if (t1 && !tocando1) tocarCampana(NOTA1, 500);
  if (t2 && !tocando2) tocarCampana(NOTA2, 500);

  tocando1 = t1;
  tocando2 = t2;

  delay(20);
}
```

## Cómo funciona
- Cada cable tiene su propia nota y suena una **campanita al tocarlo**, una vez por toque. Para que vuelva a sonar hay que soltar y tocar de nuevo.
