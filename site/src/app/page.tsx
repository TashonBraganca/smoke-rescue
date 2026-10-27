import Nav from "@/components/Nav";
import Story from "@/sections/Story";
import Mission from "@/sections/Mission";
import Smoke from "@/sections/Smoke";
import Team from "@/sections/Team";
import House from "@/sections/House";
import Built from "@/sections/Built";
import { SlamMoment } from "@/sections/Proof";
import Results from "@/sections/Results";
import Footer from "@/sections/Footer";

export default function Home() {
  return (
    <main>
      <Nav />
      <Story />
      <Mission />
      <Smoke />
      <Team />
      <House />
      <SlamMoment />
      <Results />
      <Built />
      <Footer />
    </main>
  );
}
