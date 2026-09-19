import { AuthForm } from "@/components/AuthForm";
import { GuestGuard } from "@/components/GuestGuard";

export default function SignupPage() {
  return (
    <GuestGuard>
      <AuthForm mode="signup" />
    </GuestGuard>
  );
}
