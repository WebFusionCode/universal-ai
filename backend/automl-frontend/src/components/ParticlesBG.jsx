import Particles from "react-tsparticles";

export default function ParticlesBG() {
  return (
    <Particles
      options={{
        background: { color: "#0b0f14" },
        particles: {
          number: { value: 60 },
          size: { value: 2 },
          move: { speed: 1 },
          links: {
            enable: true,
            distance: 120,
            color: "#00F5FF",
            opacity: 0.3,
          },
        },
      }}
      className="absolute top-0 left-0 w-full h-full -z-10"
    />
  );
}