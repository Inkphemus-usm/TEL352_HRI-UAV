import time
from pymavlink import mavutil

## ⚙️ Configuración

# Puerto serial de la Raspberry Pi (cambia si es necesario)
# Comúnmente es /dev/ttyACM0 para USB o /dev/ttyS0 para UART/GPIO
SERIAL_PORT = '/dev/ttyS0' 
# Velocidad de comunicación en baudios
BAUD_RATE = 115200
# Valor de throttle para el test (0-1000, siendo 1000 el máximo)
# 7% de 1000 es 70
TEST_THROTTLE = 70 
# Duración del test por motor en segundos
TEST_DURATION = 3
# Número máximo de iteraciones
MAX_ITERATIONS = 4

## 🚀 Inicialización y Conexión

def initialize_connection(port, baud):
    """Establece la conexión MAVLink a través del puerto serial."""
    print(f"✅ Intentando conectar a: {port} con {baud} baudios...")
    try:
        # Crea la conexión MAVLink. 'serial' indica que es una conexión serial.
        master = mavutil.mavlink_connection(port, baud=baud)
        # Espera hasta recibir el primer latido (heartbeat)
        master.wait_heartbeat()
        print(f"🔗 Conexión MAVLink establecida. Sistema {master.target_system}, Componente {master.target_component}")
        return master
    except Exception as e:
        print(f"❌ Error al conectar: {e}")
        return None

def arm_vehicle(master):
    """Arma el vehículo (habilita los motores)."""
    print("🚦 Armando el vehículo...")
    master.mav.command_long_send(
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 
        0, 
        1,  # 1 para armar
        0, 0, 0, 0, 0, 0
    )
    # Espera un momento para que el comando surta efecto
    time.sleep(2)
    print("✅ Vehículo armado (espera verificación en el dron).")


def disarm_vehicle(master):
    """Desarma el vehículo (deshabilita los motores)."""
    print("🛑 Desarmando el vehículo...")
    master.mav.command_long_send(
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 
        0, 
        0,  # 0 para desarmar
        0, 0, 0, 0, 0, 0
    )
    time.sleep(2)
    print("✅ Vehículo desarmado.")

def set_motor_throttle(master, motor_number, throttle_value):
    """
    Establece el valor de throttle para un motor específico usando la
    MAVLink message MOTOR_TEST (MAV_CMD_DO_MOTOR_TEST).
    """
    # El motor_number es 1-based (1, 2, 3, 4, etc.)
    
    # Envía el comando MAV_CMD_DO_MOTOR_TEST
    master.mav.command_long_send(
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, 
        0, 
        motor_number,  # Motor a testear (param1)
        mavutil.mavlink.MOTOR_TEST_THROTTLE_PWM, # Tipo de throttle (param2): PWM o ESC_INPUT (para ArduPilot/PX4)
        throttle_value, # Valor de throttle (param3). Se usa como PWM o porcentaje * 10 (0-1000)
        TEST_DURATION, # Duración en segundos (param4)
        0, 0, 0 # Parámetros restantes
    )
    print(f"   -> Motor {motor_number}: Activado a {throttle_value} (7%) por {TEST_DURATION}s...")


def run_motor_test(master):
    """Ejecuta el ciclo de test de motores."""
    
    # Los drones Quad-X tienen 4 motores
    motors = [1, 2, 3, 4] 
    
    print("\n--- 🔧 Iniciando Test de Motores ---")
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n*** 🔄 Iteración {iteration} de {MAX_ITERATIONS} ***")
        
        for motor in motors:
            # 1. Armar el vehículo antes de cada prueba de motor para asegurar que está listo (algunos firmwares lo requieren)
            # En ArduPilot, el comando DO_MOTOR_TEST no necesita que el dron esté "Armado"
            # y se recomienda no Armar para este test de seguridad, 
            # pero es necesario que el Dron esté en un Modo de Vuelo Específico (como **Stabilize** o **Manual**)
            # o que el parámetro de seguridad de pre-armado (ARMING_CHECK) lo permita.
            # NO NECESITAMOS ARM_VEHICLE() AQUÍ, SÓLO ENVIAR EL COMANDO:
            
            set_motor_throttle(master, motor, TEST_THROTTLE)
            time.sleep(TEST_DURATION)
            
            # Una vez que el tiempo de TEST_DURATION en el comando termina,
            # el motor se detiene automáticamente. No es necesario enviar un comando para pararlo.
        
        # Opcional: Pausa entre iteraciones
        time.sleep(1)

    print("\n--- ✅ Test de Motores Finalizado ---")

## 🏁 Función Principal

if __name__ == '__main__':
    master = initialize_connection(SERIAL_PORT, BAUD_RATE)
    
    if master:
        try:
            # Armar el vehículo puede ser necesario si el comando MOTOR_TEST falla.
            # En entornos seguros, **NO** armes el dron si no es estrictamente necesario.
            # Por simplicidad y seguridad, el script solo enviará el comando de test.
            
            # Si el comando no funciona, intenta deshabilitar ARMING_CHECK en ArduPilot
            # y/o intenta armar el dron antes de la prueba.
            
            run_motor_test(master)
            
        except KeyboardInterrupt:
            print("\n🚨 Script interrumpido por el usuario.")
        except Exception as e:
            print(f"\n❌ Ocurrió un error durante la ejecución: {e}")
            
        finally:
            # Puedes considerar desarmar el vehículo aquí si lo armaste previamente.
            # disarm_vehicle(master) 
            print("🛑 Desconexión o fin de script. Asegúrate de que los motores estén parados.")