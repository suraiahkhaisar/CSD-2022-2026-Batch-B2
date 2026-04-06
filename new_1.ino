#define ENA 9
#define ENB 10
#define IN1 2
#define IN2 3
#define IN3 4
#define IN4 5
#define TRIG 11
#define ECHO 12

#define F_DIR LOW
#define B_DIR HIGH

#define SPEED_BASE 80
#define SPEED_TURN 90
#define OBSTACLE_DIST 15

bool robotEnabled = false;
char currentCommand = 'F';

// 🔥 FILTER VARIABLES
int stableDistance = 999;
int obstacleCount = 0;

// ---------- MOTOR ----------
void motorL(int speed, int dir) {
  analogWrite(ENA, speed);
  digitalWrite(IN1, dir);
  digitalWrite(IN2, !dir);
}

void motorR(int speed, int dir) {
  analogWrite(ENB, speed);
  digitalWrite(IN3, !dir);
  digitalWrite(IN4, dir);
}

// ---------- MOVEMENT ----------
void moveForward() {
  motorL(SPEED_BASE, F_DIR);
  motorR(SPEED_BASE, F_DIR);
}

void turnLeft() {
  motorL(40, F_DIR);
  motorR(SPEED_TURN, F_DIR);
}

void turnRight() {
  motorL(SPEED_TURN, F_DIR);
  motorR(40, F_DIR);
}

void stopRobot() {
  motorL(0, F_DIR);
  motorR(0, F_DIR);
}

// ---------- ULTRASONIC FILTER ----------
int getFilteredDistance() {
  int readings[5];

  for (int i = 0; i < 5; i++) {
    digitalWrite(TRIG, LOW); delayMicroseconds(2);
    digitalWrite(TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(TRIG, LOW);

    long duration = pulseIn(ECHO, HIGH, 20000);
    int d = (duration == 0) ? 999 : duration * 0.034 / 2;

    readings[i] = d;
    delay(5);
  }

  // Take minimum (safer for obstacles)
  int minVal = readings[0];
  for (int i = 1; i < 5; i++) {
    if (readings[i] < minVal) minVal = readings[i];
  }

  return minVal;
}

void setup() {
  Serial.begin(9600);

  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);

  stopRobot();
  Serial.println("ARDUINO_READY");
}

void loop() {

  // ---------- SERIAL ----------
  if (Serial.available()) {
    char cmd = Serial.read();

    if (cmd == 'S') {
      robotEnabled = true;
    } 
    else if (cmd == 'X') {
      robotEnabled = false;
      stopRobot();
    } 
    else if (robotEnabled) {
      currentCommand = cmd;
    }
  }

  if (!robotEnabled) return;

  // ---------- FILTERED OBSTACLE ----------
  int d = getFilteredDistance();

  if (d > 2 && d < OBSTACLE_DIST) {
    obstacleCount++;
  } else {
    obstacleCount = 0;
  }

  // 🔥 ONLY trigger if stable obstacle
  if (obstacleCount > 3) {
    stopRobot();

    motorL(120, B_DIR);
    motorR(120, B_DIR);
    delay(200);

    stopRobot();
    Serial.println("EVENT:OBSTACLE");

    obstacleCount = 0;
    return;
  }

  // ---------- MOVEMENT ----------
  if (currentCommand == 'F') moveForward();
  else if (currentCommand == 'L') turnLeft();
  else if (currentCommand == 'R') turnRight();
  else moveForward();

  delay(20); // 🔥 smooth loop
}