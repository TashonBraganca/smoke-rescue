import type { Metadata } from 'next';
import Rescue from '@/rescue/Rescue';
export const metadata: Metadata = {
  title: 'Run your own rescue | Smoke Rescue',
  description: 'Choose one of five houses and send in one, two or three robot dogs. Watch a recorded EASY simulation begin at once.',
};
export default function RescuePage() { return <Rescue />; }
